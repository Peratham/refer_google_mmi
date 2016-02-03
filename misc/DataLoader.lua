require 'hdf5'
local utils = require 'misc.utils'

-- Read json file has
-- 0. 'refs': list of {ref_id, ann_id, split, image_id, category_id, sent_ids}
-- 1. 'images': list of {image_id, ref_ids, file_name, width, height, h5_id}
-- 2. 'sentences': list of {sent_id, tokens, h5_id}
-- 3. 'anns': list of {ann_id, image_id, category_id, bbox}
-- 4. 'ix_to_word' 
-- 5. 'word_to_ix'
-- Read h5 file has 
-- 0. /images
-- 1. /labels

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  self.ix_to_word = self.info.ix_to_word
  self.vocab_size = utils.count_keys(self.ix_to_word)
  print('vocab size is ' .. self.vocab_size)
  
  -- Construct Refs, Images, Sentences, and Anns
  local Refs = {}
  for i = 1, #self.info.refs do
    local ref = self.info.refs[i]
    Refs[ref['ref_id']] = ref
  end

  local Images = {}
  for i = 1, #self.info.images do
    local image = self.info.images[i]
    Images[image['image_id']] = image
  end

  local Sentences = {}
  for i = 1, #self.info.sentences do
    local sent = self.info.sentences[i]
    Sentences[sent['sent_id']] = sent
  end

  local Anns = {}
  for i = 1, #self.info.anns do
    local ann = self.info.anns[i]
    Anns[ann['ann_id']] = ann
  end

  self.Refs, self.Images, self.Sentences, self.Anns = Refs, Images, Sentences, Anns

  -- open the hdf5 file
  print('DataLoader loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  
  -- extract image size from dataset
  local images_size = self.h5_file:read('/images'):dataspaceSize()
  assert(#images_size == 4, '/images should be a 4D tensor')
  assert(images_size[3] == images_size[4], 'width and height must match')
  self.num_images = images_size[1]
  self.num_channels = images_size[2]
  self.max_image_size = images_size[3]
  print(string.format('read %d images of size %dx%dx%d', self.num_images, 
            self.num_channels, self.max_image_size, self.max_image_size))

  -- load in the sequence data
  local seq_size = self.h5_file:read('/labels'):dataspaceSize()
  self.seq_length = seq_size[2]
  print('max sequence length in data is ' .. self.seq_length)

  -- separate out indexes for each of the provided splits 
  self.split_ix = {}   -- e.g., split_ix['val'] = {ref_id[1], ref_id[3], ref_id[12], ...} 
  self.iterators = {}  -- e.g., iterators['val'] = 2 --> point to ref_id[3]
  for ref_id, ref in pairs(self.Refs) do
    local split = ref['split']
    if not self.split_ix[split] then
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.split_ix[split], ref_id)
  end

  for k, v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
  end
end

function DataLoader:shuffle(split)
  -- shuffle ref_ids within the table of self.split_ix[split]
  local iterations = #self.split_ix[split]
  local json
  for i = iterations, 2, -1 do
    j = math.random(i)
    self.split_ix[split][i], self.split_ix[split][j] = self.split_ix[split][j], self.split_ix[split][i]
  end
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:getSeqLength()
  return self.seq_length
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - X (N,3,H,W) containing the global images
  - y (L,M) containing the captions as columns (which is better for contiguous memory during training)
  - info table of length N, containing ref_id, pos_bbox, and sampled neg_bbox
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getPosNegBatch(opt)
  local split = utils.getopt(opt, 'split', 'train')  -- default is training
  local batch_size = utils.getopt(opt, 'batch_size', 5)
  local seq_per_img = utils.getopt(opt, 'seq_per_img', 3)

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pick and index of the datapoint to load next
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
  local label_batch = torch.LongTensor(batch_size * seq_per_img, self.seq_length)
  local max_index = #split_ix
  local wrapped = false
  local infos = {}
  for i=1, batch_size do

    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterators[split] = ri_next

    -- fetch the referred object
    local ref_id = split_ix[ri]
    local ref = self.Refs[ref_id]
    local image = self.Images[ref['image_id']]
    local category_id = ref['category_id']

    -- its bounding box
    local x, y, w, h = unpack(ref['bbox'])
    local nx, ny, nw, nh = math.floor(x/image['width']*self.max_image_size)+1, math.floor(y/image['height']*self.max_image_size)+1, 
    math.max(math.floor(w/image['width']*self.max_image_size), 1), math.max(math.floor(h/image['height']*self.max_image_size), 1)

    local pos_bbox = {nx, ny, nw, nh}

    -- the negative region (bbox)
    local ann_ids = image['ann_ids']
    local shuffle = torch.randperm(#ann_ids) -- shuffle the ann_ids
    local neg_ann
    -- 1) choose ann of same category_id as pos_ref
    for i = 1, #ann_ids do
      local cand_ann_id = ann_ids[shuffle[i]]
      local cand_ann = self.Anns[cand_ann_id]
      if cand_ann_id ~= ref['ann_id'] and cand_ann['category_id'] == ref['category_id'] then
        neg_ann = cand_ann
        break
      end
    end
    -- 2) choose ann within the same image if 1) failed
    if neg_ann == nil then
      for i = 1, #ann_ids do
        local cand_ann_id = ann_ids[shuffle[i]]  -- randome choose one
        local cand_ann = self.Anns[cand_ann_id]
        if cand_ann_id ~= ref['ann_id'] then
          neg_ann = cand_ann
          break
        end
      end
    end
    -- 3) choose random region if 1) and 2) failed
    if neg_ann ~= nil then
      x, y, w, h = unpack(neg_ann['bbox'])
    else
      x, y, w, h = torch.random(64), torch.random(64) , 64, 64
    end
    nx, ny, nw, nh = math.max(math.floor(x/image['width']*self.max_image_size),1), math.max(math.floor(y/image['height']*self.max_image_size), 1), 
    math.max(math.floor(w/image['width']*self.max_image_size), 1), math.max(math.floor(h/image['height']*self.max_image_size), 1)
    -- nx, ny, nw, nh = math.max(math.floor(x/image['width']*self.max_image_size),1), math.max(math.floor(y/image['height']*self.max_image_size), 1), 
    -- math.max(math.floor(w/image['width']*self.max_image_size), 1), math.max(math.floor(h/image['height']*self.max_image_size), 1)
    local neg_bbox = {nx, ny, nw, nh}

    -- fetch the images from h5
    local h5_id = image['h5_id']
    local img = self.h5_file:read('/images'):partial({h5_id,h5_id},{1,self.num_channels},{1,self.max_image_size},{1,self.max_image_size})
    img_batch_raw[i] = img

    -- fetch the sequence labels
    local sent_ids = ref['sent_ids']
    local sent_id
    local seq = torch.LongTensor(seq_per_img, self.seq_length)
    for q=1, seq_per_img do
      if #sent_ids < seq_per_img then
        sent_id = sent_ids[torch.random(1, #sent_ids)]
      else
        sent_id = sent_ids[q]
      end 
      h5_id = self.Sentences[sent_id]['h5_id']
      seq[{ {q,q} }] = self.h5_file:read('/labels'):partial({h5_id, h5_id}, {1,self.seq_length})
    end

    local il = (i-1)*seq_per_img+1
    label_batch[{ {il,il+seq_per_img-1} }] = seq

    -- and record associated info as well
    local info_struct = {}
    -- info_struct.ref = ref    
    info_struct.ref_id = ref_id
    info_struct.pos_bbox = pos_bbox
    info_struct.neg_bbox = neg_bbox
    table.insert(infos, info_struct)
  end 

  local data = {}
  data.images = img_batch_raw
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos = infos
  return data

end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - X (N,3,H,W) containing the global images
  - Y (L,M) containing the captions as columns (which is better for contiguous memory during training)
  - info table of length N, containing ref_id and bbox
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
  local seq_per_img = utils.getopt(opt, 'seq_per_img', 3) -- number of sequences to return per image

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
  local label_batch = torch.LongTensor(batch_size * seq_per_img, self.seq_length)
  local max_index = #split_ix
  local wrapped = false
  local infos = {}
  for i=1,batch_size do

    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterators[split] = ri_next
    local ref_id = split_ix[ri]
    assert(ref_id ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)    

    -- fetch the images from h5
    local ref = self.Refs[ref_id]
    local image = self.Images[ref['image_id']]
    local x, y, w, h = unpack(ref['bbox'])
    nx, ny, nw, nh = math.max(math.floor(x/image['width']*self.max_image_size),1), math.max(math.floor(y/image['height']*self.max_image_size), 1), 
    math.max(math.floor(w/image['width']*self.max_image_size), 1), math.max(math.floor(h/image['height']*self.max_image_size), 1)
    -- local nx, ny, nw, nh = math.floor(x/image['width']*self.max_image_size)+1, math.floor(y/image['height']*self.max_image_size)+1, 
    -- math.floor(w/image['width']*self.max_image_size), math.floor(h/image['height']*self.max_image_size)

    local h5_id = image['h5_id']
    local img = self.h5_file:read('/images'):partial({h5_id,h5_id},{1,self.num_channels},{1,self.max_image_size},{1,self.max_image_size})
    img_batch_raw[i] = img

    -- fetch the sequence labels
    local sent_ids = ref['sent_ids']
    local sent_id
    local seq = torch.LongTensor(seq_per_img, self.seq_length)
    for q=1, seq_per_img do
      if #sent_ids < seq_per_img then
        sent_id = sent_ids[torch.random(1, #sent_ids)]
      else
        sent_id = sent_ids[q]
      end 
      h5_id = self.Sentences[sent_id]['h5_id']
      seq[{ {q,q} }] = self.h5_file:read('/labels'):partial({h5_id, h5_id}, {1,self.seq_length})
    end

    local il = (i-1)*seq_per_img+1
    label_batch[{ {il,il+seq_per_img-1} }] = seq

    -- and record associated info as well
    local info_struct = {}
    -- info_struct.ref = ref    
    info_struct.ref_id = ref_id
    info_struct.bbox = {nx, ny, nw, nh}
    table.insert(infos, info_struct)
  end

  local data = {}
  data.images = img_batch_raw
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos = infos
  return data

end