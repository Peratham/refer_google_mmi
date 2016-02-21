require 'image'
require 'misc.basic_modules'
local utils = require 'misc.utils'
local net_utils = {}


-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local use_global = utils.getopt(opt, 'use_global', 1)
  local use_region = utils.getopt(opt, 'use_region', 1)
  local finetune = utils.getopt(opt, 'finetune', 0)

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)
    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end
    cnn_part:add(layer)
  end

  -- construct cnns
  if use_global + use_region == 1 then
    return cnn_part

  elseif use_global + use_region == 2 then
    local cnns = nn.ParallelTable()
    cnns:add(cnn_part) 
    -- gcnn and rcnn are shared if no finetune is needed
    -- if finetune == 1 then cnns:add(cnn_part:clone()) else cnns:add(cnn_part) end 
    cnns:add(cnn_part:clone())
    return cnns

  else
    error('Cannot construct CNN(s) with use_global = 0 and use_region = 0.')
  end
end


-- joint embedding layer, you can choose what to be combined from the cnn outputs
function net_utils.build_jemb(cnn, opt)
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  local dropout = utils.getopt(opt, 'dropout', 0.5)
  local loc_weight = utils.getopt(opt, 'loc_weight', 10)
  local use_global = utils.getopt(opt, 'use_global', 1)
  local use_region = utils.getopt(opt, 'use_region', 1)
  assert(use_global+use_region > 0, 'use at least one of global and region images')

  local jemb_part = nn.Sequential()
  jemb_part:add(nn.FlattenTable())  -- flattern input, e.g., {{gfeats, rfeats}, lfeats}
  local d = 0
  local fc = cnn:get(39)

  -- M is Parallel Table
  local M = nn.ParallelTable()
  -- add 4096-->1000 layer for gcnn
  if use_global > 0 then 
    M:add(fc:clone())
    d = d + 1000
  end
  -- add 4096-->1000 layer for rcnn
  if use_region > 0 then
    d = d + 1000
    M:add(fc:clone())
  end
  -- add 5-D location features
  -- M:add(nn.Identity())
  M:add(nn.Scale(loc_weight))
  d = d + 5
  jemb_part:add(M)

  -- jointly encode them
  jemb_part:add(nn.JoinTable(2))
  jemb_part:add(nn.Linear(d, encoding_size))
  jemb_part:add(cudnn.ReLU(true))
  jemb_part:add(nn.Dropout(dropout))

  return jemb_part
end


-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
function net_utils.prepro(imgs, infos, bbox_type, data_augment, on_gpu)
  -- imgs: N*3*256*256
  -- infos: table of {ref_id, bbox (new)}
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local batch_size, h, w = imgs:size(1), imgs:size(3), imgs:size(4)
  local cnn_input_size = 224

  -- cropping data augmentation, if needed
  local gimgs = torch.zeros(batch_size, 3, cnn_input_size, cnn_input_size)
  local rimgs = torch.zeros(batch_size, 3, cnn_input_size, cnn_input_size)
  local lfeats = torch.zeros(batch_size, 5)

  local xoff, yoff
  local nx, ny, nw, nh
  for i = 1, batch_size do
    nx, ny, nw, nh = unpack(infos[i][bbox_type])
    if data_augment then
      xoff = torch.random( math.max(1, nx+math.ceil(nw/2)-223), math.min(nx, 33) )
      yoff = torch.random( math.max(1, ny+math.ceil(nh/2)-223), math.min(ny, 33) )
      nw, nh = math.min(nw, xoff+224-nx), math.min(nh, yoff+224-ny)
      -- crop global image
      gimgs[i] = imgs[{ i, {}, {yoff, yoff+cnn_input_size-1}, {xoff, xoff+cnn_input_size-1} }]
      -- crop and scale region image
      rimgs[i] = image.scale( imgs[{ i, {}, {ny, ny+nh-1}, {nx, nx+nw-1} }], cnn_input_size, cnn_input_size)
      -- compute 5-D location feature
      lfeats[i] = torch.FloatTensor{nx-xoff, ny-yoff, nx-xoff+nw, ny-yoff+nh, nh*nw/cnn_input_size} / cnn_input_size -- lazily ignore +1 or -1
    else
      gimgs[i] = image.scale(imgs[i], cnn_input_size, cnn_input_size)
      rimgs[i] = image.scale(imgs[{ i, {}, {ny, ny+nh-1}, {nx, nx+nw-1} }], cnn_input_size, cnn_input_size)
      lfeats[i] = torch.FloatTensor{nx, ny, nx+nw, ny+nh, nh*nw/cnn_input_size} / cnn_input_size 
    end
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then 
    gimgs = gimgs:cuda(); rimgs = rimgs:cuda(); lfeats = lfeats:cuda()
  else 
    gimgs = gimgs:float(); rimgs = rimgs:float(); lfeats = lfeats:float()
  end

  -- subtract vgg mean
  net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(gimgs) -- a noop if the types match

  gimgs:add(-1, net_utils.vgg_mean:expandAs(gimgs))
  rimgs:add(-1, net_utils.vgg_mean:expandAs(rimgs))

  return gimgs, rimgs, lfeats
end

function net_utils.combine(gimgs, rimgs, opt)
  -- use gimgs or rimgs according to use_global and use_region
  local use_global = utils.getopt(opt, 'use_global', 1)
  local use_region = utils.getopt(opt, 'use_region', 1)
  local x
  if opt.use_global + opt.use_region == 2 then
    x = {gimgs, rimgs}
  elseif opt.use_global == 1 then
    x = gimgs
  elseif opt.use_region == 1 then
    x = rimgs
  else
    error('Use at least one of gimgs and rimgs')
  end
  return x
end


function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end


function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- -- hiding this piece of code on the bottom of the file, in hopes that
-- -- noone will ever find it. Lets just pretend it doesn't exist
-- function net_utils.language_eval(predictions, dataset, id, split)
--   -- this is gross, but we have to call coco python code.
--   -- Not my favorite kind of thing, but here we go
--   local val_path = path.join('./valtest', dataset, 'model_id' .. id .. '_' .. split)
--   local out_struct = {predictions = predictions}
--   utils.write_json(val_path .. '.json', out_struct) -- serialize to json (ew, so gross)
--   os.execute('python ./misc/python_eval.py ' .. '--dataset_splitBy_click ' .. dataset .. ' --model_id ' .. id .. ' --split ' .. split) -- i'm dying over here
--   local result_struct = utils.read_json(val_path .. '_out.json') -- god forgive me
--   return result_struct['overall']
-- end

function net_utils.language_eval(predictions, dataset, id, split)
  local cache_lang_dataset_dir = path.join('./cache/lang', dataset)
  -- we don't check isdir here...
  -- otherwise we have to luarocks install some other packages, e.g., posix, luafilesystem
  os.execute('mkdir '..cache_lang_dataset_dir)  

  local cache_path = path.join(cache_lang_dataset_dir, 'model_id' .. id .. '_' .. split .. '.json')
  utils.write_json(cache_path, {predictions = predictions})
  -- call python to evaluate each sent with ground-truth sentences
  os.execute('python ./misc/python_eval_lang.py ' .. '--dataset_splitBy ' .. dataset .. ' --model_id ' .. id .. ' --split ' .. split)
  -- return results
  local out_path = path.join(cache_lang_dataset_dir, 'model_id' .. id .. '_' .. split .. '_out.json')
  local out = utils.read_json(out_path)
  local result_struct = out['overall']  -- overall scores
  return result_struct
end

return net_utils
