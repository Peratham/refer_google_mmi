
require 'torch'
require 'nn'
require 'cudnn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test an Image Captioning model')
cmd:text()
cmd:text('Options')
-- Input paths
cmd:option('-dataset', 'refcoco_licheng', 'name of our dataset+splitBy')
cmd:option('-id', 0, 'model id to be evaluated')   -- corresponding to opt.id in train.lua     
-- Basic options
cmd:option('-batch_size', 8, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
-- For evaluation on refer dataset for some split
cmd:option('-split', 'testA', 'what split to use: val|test|train')
-- misc
cmd:option('-seed', 24, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.dataset) > 0 and string.len(opt.id) > 0, 'must provide dataset name and model id')
model_path = path.join('model', opt.dataset, 'model_id' .. opt.id ..'.t7')
local checkpoint = torch.load(model_path)

-- override and collect parameters
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'input_encoding_size', 'cnn_proto', 'cnn_model', 'seq_per_img', 'jemb_use_global',
'jemb_use_region'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
local vocab = checkpoint.vocab -- ix -> word mapping

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local input_h5   = 'cache/data/' .. opt.dataset .. '/data.h5'    -- path to the h5file containing the preprocessed dataset
local input_json = 'cache/data/' .. opt.dataset .. '/data.json'  -- path to the json file containing additional info and vocab
local loader
loader = DataLoader{h5_file = input_h5, json_file = input_json}

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.crit = nn.LanguageModelCriterion()
protos.lm:createClones() -- reconstruct clones inside the language model
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local num_images = utils.getopt(evalopt, 'num_images', true)

  protos.cnn:evaluate()
  protos.jemb:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  while true do

    -- fetch a batch of data
    local data = loader:getBatch({batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img})
    local gimgs, rimgs, lfeats = net_utils.prepro(data.images, data.infos, 'bbox', false, opt.gpuid >= 0)
    local x = net_utils.combine(gimgs, rimgs, {use_global=opt.jemb_use_global, use_region=opt.jemb_use_region})  -- combine features accoring to use_global/region
    n = n + gimgs:size(1)

    -- forward the model to get loss
    local cnn_feats = protos.cnn:forward(x)
    local jemb_feats = protos.jemb:forward{cnn_feats, lfeats}
    local expanded_feats = protos.expander:forward(jemb_feats)
    local logprobs = protos.lm:forward{expanded_feats, data.labels}
    local loss = protos.crit:forward(logprobs, data.labels)

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    local sample_opts = { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
    local seq = protos.lm:sample(jemb_feats, sample_opts)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {ref_id = data.infos[k].ref_id, sent = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('ref_id%s: %s', entry.ref_id, entry.sent))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, num_images)
    if verbose then
      print(string.format('evaluating performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if num_images >= 0 and n >= num_images then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.dataset, opt.id, split)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

local loss, split_predictions, lang_stats = eval_split(opt.split, {num_images = opt.num_images})
print('loss: ', loss)
if lang_stats then
  print(lang_stats)
end

-- write HTML, feed in dataset, model_id, split
-- saved in vis/dataset/
-- os.execute('python vis.py ' .. '--dataset_splitBy_click ' .. opt.dataset .. ' --model_id ' .. opt.id .. ' --split ' .. opt.split) 
os.execute('python vis_lang.py ' .. '--dataset_splitBy ' .. opt.dataset .. ' --model_id ' .. opt.id .. ' --split ' .. opt.split) 
