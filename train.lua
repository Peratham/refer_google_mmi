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
local basic_utils = require 'misc.basic_modules'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'


-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-dataset', 'refcoco_licheng', 'name of our our dataset+splitBy')
cmd:option('-cnn_proto','model/vgg/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','model/vgg/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')

-- Parameter on Margin Ranking Criterion
cmd:option('-ranking_weight', 2, 'the weight on ranking loss')
cmd:option('-ranking_margin', 1, 'the margin in the ranking loss')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',8,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-seq_per_img',3,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 20000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the Language Model
cmd:option('-lm_optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-lm_drop_out', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-lm_learning_rate',4e-4,'learning rate')
cmd:option('-lm_optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-lm_optim_beta',0.999,'beta used for adam')

-- Optimization: for the Joint Embedding
cmd:option('-finetune_jemb_after', 20000, 'After what iteration do we start finetuning the joint embedding? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-jemb_use_global', 1, 'joint embedding global context')
cmd:option('-jemb_use_region', 1, 'joint embedding local region')
cmd:option('-jemb_drop_out', 0.5, 'strength of dropout in the joint embeddingig')
cmd:option('-jemb_optim','adam','optimization to use for joint embedding')
cmd:option('-jemb_optim_alpha',0.8,'alpha for momentum of joint embedding')
cmd:option('-jemb_optim_beta',0.999,'alpha for momentum of joint embedding')
cmd:option('-jemb_learning_rate',1e-5,'learning rate for the joint embedding')
cmd:option('-jemb_weight_decay', 0, 'L2 weight decay just for the joint embedding')
cmd:option('-jemb_loc_weight', 10, 'weight on 5-D location features')

-- Optimization: for the CNN
cmd:option('-cnn_finetune', 0, 'if no let\'s share the cnn parameters for saving memory')
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 0, 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 8, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

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
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local input_h5   = './cache/data/' .. opt.dataset .. '/data.h5'    -- path to the h5file containing the preprocessed dataset
local input_json = './cache/data/' .. opt.dataset .. '/data.json'  -- path to the json file containing additional info and vocab
local loader = DataLoader{h5_file = input_h5, json_file = input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}

if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  -- prepare cnn
  net_utils.unsanitize_gradients(protos.cnn)
  -- prepare jemb
  net_utils.unsanitize_gradients(protos.jemb)
  -- prepare LM
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  -- prepare jemb expander, not in checkpoints, create manually
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- prepare feed layer, feeding features to two loss functions
  protos.feeder = basic_utils.FeedToCrits(opt.ranking_weight)
  -- prepare criterion1 for the language model, not in checkpoints, create manually
  protos.crit1 = nn.LanguageModelCriterion()
  protos.crit2 = nn.LanguageRankingCriterion(opt.ranking_margin)
else
  -- initialize the ConvNet
  local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, 'cudnn')
  -- create protos for language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.lm_drop_out
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  protos.lm = nn.LanguageModel(lmOpt)
  -- create protos for joint embedding model
  local jembOpt = {}
  jembOpt.use_global = opt.jemb_use_global
  jembOpt.use_region = opt.jemb_use_region
  jembOpt.loc_weight = opt.jemb_loc_weight
  jembOpt.input_encoding_size = opt.input_encoding_size
  jembOpt.dropout = opt.jemb_drop_out
  protos.jemb = net_utils.build_jemb(cnn_raw, {use_global=jembOpt.use_global, use_region=jembOpt.use_region, loc_weight=jembOpt.loc_weight, 
  	encoding_size=jembOpt.input_encoding_size, dropout=jembOpt.dropout})
  -- create cnn for global and region image
  protos.cnn = net_utils.build_cnn(cnn_raw, {finetune = opt.cnn_finetune, use_global=jembOpt.use_global, use_region=jembOpt.use_region})
  -- jemb expander
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- prepare feed layer, feeding features to two loss functions
  protos.feeder = basic_utils.FeedToCrits(opt.ranking_weight)
  -- criterion for the language model
  protos.crit1 = nn.LanguageModelCriterion()
  -- criterion for the ranking loss
  protos.crit2 = nn.LanguageRankingCriterion(opt.ranking_margin)
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local lm_params, lm_grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
local jemb_params, jemb_grad_params = protos.jemb:getParameters()
print('total number of parameters in LM: ', lm_params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
print('total number of parameters in JEMB: ', jemb_params:nElement())
assert(lm_params:nElement() == lm_grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())
assert(jemb_params:nElement() == jemb_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')
local thin_jemb = protos.jemb:clone('weight', 'bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
net_utils.sanitize_gradients(thin_jemb)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
	local verbose = utils.getopt(evalopt, 'verbose', true)
	local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.cnn:evaluate()
	protos.jemb:evaluate()
	protos.lm:evaluate()
	loader:resetIterator(split)
	local n = 0
	local loss_sum = 0
	local loss_evals = 0
	local predictions = {}
	local vocab = loader:getVocab()
	while true do
		-- fetch a batch of data
    local data = loader:getPosNegBatch({batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img})
		local gimgs1, rimgs1, lfeats1 = net_utils.prepro(data.images, data.infos, 'pos_bbox', false, opt.gpuid >= 0)
		local gimgs2, rimgs2, lfeats2 = net_utils.prepro(data.images, data.infos, 'neg_bbox', false, opt.gpuid >= 0)
		local batch_size = gimgs1:size(1)

		-- combine positive pair and negative pair
		local Gimgs = torch.cat(gimgs1, gimgs2, 1)
		local Rimgs = torch.cat(rimgs1, rimgs2, 1)
		local Lfeats = torch.cat(lfeats1, lfeats2, 1)
		local Labels = torch.cat(data.labels, data.labels)  -- [data.labels | data.labels]
    local x = net_utils.combine(Gimgs, Rimgs, {use_global=opt.jemb_use_global, use_region=opt.jemb_use_region})  -- combine features accoring to use_global/region
		n = n + batch_size

    -- forward the model to get loss
    local cnn_feats = protos.cnn:forward(x)
    local jemb_feats = protos.jemb:forward{cnn_feats, Lfeats}
    local expanded_feats = protos.expander:forward(jemb_feats)
    local logprobs = protos.lm:forward{expanded_feats, Labels}
    local fed1, fed2 = unpack(protos.feeder:forward(logprobs))
    local loss1 = protos.crit1:forward(fed1, data.labels)
    local loss2 = protos.crit2:forward(fed2, data.labels)

    local loss = (loss1 + opt.ranking_weight*loss2)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    local seq = protos.lm:sample(jemb_feats[{ {1, batch_size}, {} }])
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1, #sents do
    	local entry = {ref_id = data.infos[k].ref_id, sent = sents[k]}
    	table.insert(predictions, entry)
    	if verbose then
    		print(string.format('ref_id%s: %s', entry.ref_id, entry.sent))
    	end
    end

    -- if we wrapped around the split or used up val imgs budget than bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
    	print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if val_images_use >= 0 and n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.dataset, opt.id, split)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
-- shuffle the data
loader:shuffle('train')
local iter = 0

local function lossFun()
  -- cnn mode
  if opt.cnn_finetune > 0 then
    protos.cnn:training()
		cnn_grad_params:zero()
	else
		protos.cnn:evaluate()
	end
  -- jemb mode
  if opt.finetune_jemb_after >= 0 and iter >= opt.finetune_jemb_after then
    protos.jemb:training()
    jemb_grad_params:zero()
  else
    protos.jemb:evaluate()
  end
  -- lm mode
  protos.lm:training()
  lm_grad_params:zero()

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
	-- fetch a batch of data
	local data = loader:getPosNegBatch({batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img})
	local gimgs1, rimgs1, lfeats1 = net_utils.prepro(data.images, data.infos, 'pos_bbox', true, opt.gpuid >= 0)
	local gimgs2, rimgs2, lfeats2 = net_utils.prepro(data.images, data.infos, 'neg_bbox', true, opt.gpuid >= 0)
	local batch_size = gimgs1:size(1)

	-- combine positive pair and negative pair
	local Gimgs = torch.cat(gimgs1, gimgs2, 1)
	local Rimgs = torch.cat(rimgs1, rimgs2, 1)
	local Lfeats = torch.cat(lfeats1, lfeats2, 1)
	local Labels = torch.cat(data.labels, data.labels)  -- [data.labels | data.labels]
  local x = net_utils.combine(Gimgs, Rimgs, {use_global=opt.jemb_use_global, use_region=opt.jemb_use_region})  -- combine features accoring to use_global/region
	
  -- forward the model to get loss
  local cnn_feats = protos.cnn:forward(x)
  local jemb_feats = protos.jemb:forward{cnn_feats, Lfeats}
  local expanded_feats = protos.expander:forward(jemb_feats)
  local logprobs = protos.lm:forward{expanded_feats, Labels}
  local fed1, fed2 = unpack(protos.feeder:forward(logprobs))
  local loss1 = protos.crit1:forward(fed1, data.labels)
  local loss2 = protos.crit2:forward(fed2, data.labels)
  local loss = loss1 + opt.ranking_weight*loss2

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  local dfed2 = protos.crit2:backward(fed2, data.labels)
  local dfed1 = protos.crit1:backward(fed1, data.labels)
	local dlogprobs = protos.feeder:backward(logprobs, {dfed1, dfed2})
  local dexpanded_feats, dummy = unpack(protos.lm:backward({expanded_feats, Labels}, dlogprobs))
  if opt.finetune_jemb_after >= 0 and iter >= opt.finetune_jemb_after then
    local djemb_feats = protos.expander:backward(jemb_feats, dexpanded_feats)
    local dcnn_feats, ddummy = unpack(protos.jemb:backward({cnn_feats, Lfeats}, djemb_feats))
    if opt.cnn_finetune > 0 then
      local dx = protos.cnn:backward(x, dcnn_feats)
    end
  end

  -- clip gradients
  lm_grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization on joint embedding layer
  if opt.jemb_weight_decay > 0 then
    jemb_grad_params:add(opt.jemb_weight_decay, jemb_params)
    jemb_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  -- apply L2 regularization on CNN
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  -----------------------------------------------------------------------------
  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local jemb_optim_state = {}
local lm_optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score

while true do  

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f', iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use})
    print('validation loss: ', val_loss)
    val_loss_history[iter] = val_loss
    if lang_stats then
      print(lang_stats)
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, opt.dataset, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
        save_protos.jemb = thin_jemb
        save_protos.cnn = thin_cnn
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- decay the learning rates for LM, CNN and JEMB
  local lm_learning_rate = opt.lm_learning_rate
  local jemb_learning_rate = opt.jemb_learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    lm_learning_rate = lm_learning_rate * decay_factor
    jemb_learning_rate = jemb_learning_rate * decay_factor
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform LM update
  if opt.lm_optim == 'rmsprop' then
    rmsprop(lm_params, lm_grad_params, lm_learning_rate, opt.lm_optim_alpha, opt.optim_epsilon, lm_optim_state)
  elseif opt.lm_optim == 'sgd' then
    sgd(lm_params, lm_grad_params, opt.lm_learning_rate)
  elseif opt.lm_optim == 'adam' then
    adam(lm_params, lm_grad_params, lm_learning_rate, opt.lm_optim_alpha, opt.lm_optim_beta, opt.optim_epsilon, lm_optim_state)
  else
    error('bad option opt.lm_optim')
  end

  -- do a JEMB and CNN update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_jemb_after >= 0 and iter >= opt.finetune_jemb_after then
    if opt.jemb_optim == 'rmsprop' then
      rmsprop(jemb_params, jemb_grad_params, jemb_learning_rate, opt.jemb_optim_alpha, opt.optim_epsilon, jemb_optim_state)
    elseif opt.jemb_optim == 'sgd' then
      sgd(jemb_params, jemb_grad_params, opt.jemb_learning_rate)
    elseif opt.jemb_optim == 'adam' then
      adam(jemb_params, jemb_grad_params, jemb_learning_rate, opt.jemb_optim_alpha, opt.jemb_optim_beta, opt.optim_epsilon, jemb_optim_state)
    else
      error('bad option opt.lm_optim')
    end

    if opt.cnn_finetune > 0 then
      if opt.cnn_optim == 'sgd' then
        sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
      elseif opt.cnn_optim == 'sgdm' then
        sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
      elseif opt.cnn_optim == 'adam' then
        adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
      else
        error('bad option for opt.cnn_optim')
      end
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
