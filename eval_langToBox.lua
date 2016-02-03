--[[
Inputs:
1. imageToBoxes.json: {image_id: [bbox]}
2. sents.json: [{'sent', 'ref_id', 'tokens', 'bbox'}]
--]]
require 'torch'
require 'nn'
require 'cudnn'
require 'nngraph'
require 'hdf5'
require 'image'
-- local imports
require 'misc.BoxLoader'
require 'misc.LanguageModel'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test accuracy of click(bbox) given sentence')
cmd:text()
cmd:text('Options')
-- Input paths
cmd:option('-dataset', 'refcoco_licheng', 'name of our dataset+splitBy')
cmd:option('-id', 0, 'model id to be evaluated')
-- Basic options
cmd:option('-batch_size', 8, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_sents', -1, 'how many sentences to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-vis', -1, 'if 1 then we visualize all  boxes of each image')
-- For evaluation on refer dataset for some split
cmd:option('-boxes_type', 'gd', 'use ground-truth (gd) or predicted (pred) imageToBoxes?')
cmd:option('-sents_type', 'gd', 'use ground-truth (gd) or predicted (pred) sentences?')
cmd:option('-split', 'testA', 'what split to use: val|test|train')
-- misc
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

cmd:text()




-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test accuracy of click(bbox) given sentence')
cmd:text()
cmd:text('Options')
-- Input paths
cmd:option('-dataset', 'refcoco_licheng', 'name of our dataset+splitBy')
cmd:option('-id', 0, 'model id to be evaluated')
-- Basic options
cmd:option('-eval_model_id', 0, 'model used to evaluate others...super wierd here!')  -- don't like it at all...
cmd:option('-num_sents', -1, 'how many sentences to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-batch_size', 8, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-vis', -1, 'if 1 then we visualize boxes during evaluation')
-- Test on what split
cmd:option('-split', 'testA', 'what split to use: val|test|train')
-- Use ground-truth boxes or detected objects(proposals)
cmd:option('-boxes_type', 'gd', 'use ground-truth (gd) or predicted (pred) imageToBoxes?')
-- misc
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')  -- for CPU

if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	cutorch.setDevice(opt.gpuid + 1)  -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the "eval_model_id" model checkpoint to evaluate!!!
-------------------------------------------------------------------------------
assert(string.len(opt.dataset) > 0 and string.len(opt.id) > 0, 'must provide dataset name and model id')
model_path = path.join('model', opt.dataset, 'model_id' .. opt.eval_model_id .. '.t7')  -- Note the model_id here! Don't mess up with opt.id.
local checkpoint = torch.load(model_path)

-- override and collect parameters
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'input_encoding_size', 'jemb_use_global', 'jemb_use_region'}
for k, v in pairs(fetch) do
	opt[v] = checkpoint.opt[v]  -- copy over options from model
end

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.expander = nn.FeatExpander(1)  -- each candidate box has only one testing sentence
protos.lm:createClones()  -- reconstruct cloens inside the language model
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end

-------------------------------------------------------------------------------
-- Create the Box Loader instance
-------------------------------------------------------------------------------
local imageToBoxes_json = path.join('cache/box', opt.dataset, 'imageToBoxes_'..opt.boxes_type..'.json')
if not utils.file_exists(imageToBoxes_json) then print('Warning: please go to cache and prepare imageToBoxes_'..opt.boxes_type..'.json'); os.exit() end

-- collect sents
os.execute('python misc/collect_sents.py --dataset_splitBy '..opt.dataset..' --model_id '..opt.id..' --split '..opt.split)  -- will save sents_id0_test.json
local sents_json = path.join('cache', 'langToBox', opt.dataset, 'sents_id'..opt.id..'_'..opt.split..'.json')

local input_h5 = path.join('data', opt.dataset, 'data.h5')
local data_json = path.join('data', opt.dataset, 'data.json')
loader = BoxLoader{h5_file = input_h5, data_json = data_json, imageToBoxes_json = imageToBoxes_json, sents_json = sents_json}

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
-- convert to Evaluate mode, never forget!
protos.cnn:evaluate()
protos.jemb:evaluate()
protos.lm:evaluate()

predictions = {}
local ix = 1
while true do

	-- fetch next sent
	local sent = loader:newSent{split = opt.split}
	if sent == nil then break end -- we've used up all sents
	if opt.num_sents > 0 and ix > opt.num_sents then break end  -- we've used up opt.num_sents sents

	-- print status
	if opt.num_sents > 0 then 
		print(string.format('Processing sent(%d/%d)', ix, opt.num_sents)) 
	else 
		print(string.format('Processing sent(%d/%d)', ix, loader:getNumSents{split = opt.split}))
	end
	ix = ix+1

	-- go to box_iterator for current sent
	Losses = nil
	while true do
		-- fetch a batch of candidate boxes 
		data = loader:getBatch{batch_size = opt.batch_size}
		local gimgs, rimgs, lfeats = net_utils.prepro(data.images, data.infos, 'bbox', false, opt.gpuid>=0)
		local x = net_utils.combine(gimgs, rimgs, {use_global=opt.jemb_use_global, use_region=opt.jemb_use_region}) -- use global or region?

		-- visualize boxes?
		if opt.vis >= 0 then
			print(data.image_id)
			visualize(gimgs, rimgs)
		end

		-- forward network to compute loss for each candidate box
		local cnn_feats = protos.cnn:forward(x)
		local jemb_feats = protos.jemb:forward{cnn_feats, lfeats}
		local expanded_feats = protos.expander:forward(jemb_feats)
		local logprobs = protos.lm:forward{expanded_feats, data.labels}
		local losses = computeLosses(logprobs, data.labels)

		-- add to Losses
		if Losses == nil then Losses = losses else Losses = torch.cat(Losses, losses) end

		-- if we wrapped then bail
		local ix0, ix1 = data.bounds.it_pos_now, data.bounds.it_max
		print(string.format(' box_%d/%d.', ix0-1, ix1))
		if data.bounds.wrapped then break end
	end

	-- choose the box with min loss
	local ls, bix = torch.min(Losses, 1)
	local pred_bbox = loader:getBoxes(sent)[bix[1]]
	assert(pred_bbox ~= nil, 'nil pred_boxs for sent_'..tostring(ix-1)..' detected.')
	sent['pred_bbox'] = pred_bbox
	sent['pred_loss'] = ls[1]
	table.insert(predictions, sent)

end

-- save
local cache_box_dataset_dir = path.join('cache/langToBox', opt.dataset)
local out_path = path.join(cache_box_dataset_dir, 'model_id'..opt.id..'_'..opt.split..'_boxes('..opt.boxes_type..')_sents(id'..opt.id..').json')
utils.write_json(out_path, {predictions=predictions})

-- print this is evaluated by model id?
print('\nEvaluated by model_id: '..opt.eval_model_id)

-- evaluate
os.execute('python misc/python_eval_box.py --dataset_splitBy '..opt.dataset..' --model_id '..opt.id..' --split '..opt.split
	..' --boxes_type '..opt.boxes_type..' --sents_type id'..opt.id)

