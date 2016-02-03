require 'hdf5'
local utils = require 'misc.utils'

--[[
Here are the read json files:
data.json has
-- 0. 'refs': list of {ref_id, ann_id, split, image_id, category_id, sent_ids}
-- 1. 'images': list of {image_id, ref_ids, file_name, width, height, h5_id}
-- 2. 'sentences': list of {sent_id, tokens, h5_id}
-- 3. 'anns': list of {ann_id, image_id, category_id, bbox}
-- 4. 'ix_to_word' 
-- 5. 'word_to_ix'
imageToBoxes.json has
-- 0. list of {'image_id', 'bbox'}
sents.json has
-- 0. list of {'sent', 'image_id', 'ref_id', 'tokens', 'bbox', 'split'}
Read h5 file has 
-- 0. /images
-- 1. /labels
--]]

local BoxLoader = torch.class('BoxLoader')

function BoxLoader:__init(opt)
	-- 1. load data.json
	print('1. BoxLoader loading json file: ', opt.data_json)
	self.info = utils.read_json(opt.data_json)
	self.ix_to_word = self.info.ix_to_word
	self.word_to_ix = self.info.word_to_ix

	-- Construct Refs and Images
	local Images = {}
	for i = 1, #self.info.images do
		local image = self.info.images[i]
		Images[image['image_id']] = image
	end
	self.Images = Images

	-- 2. load imageToBoxes.json 
	print('2. BoxLoader loading json file: ', opt.imageToBoxes_json)
	self.imageToBoxes = utils.read_json(opt.imageToBoxes_json) -- Note, image_id in string format

	-- 3. load the sents.json
	print('3. BoxLoader loading json file: ', opt.sents_json)
	self.sents = utils.read_json(opt.sents_json)

	-- 4. open the hdf5 file
	print('4. BoxLoader loading h5_file: ', opt.h5_file)
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

	-- extract sequence size
	self.seq_length = self.h5_file:read('/labels'):dataspaceSize()[2]
	print('max sequence length of labels is ' .. self.seq_length)

	-- print vocab size
	print('vocab size is ' .. utils.count_keys(self.ix_to_word))

	-- separate out indexes for each of the provided splits
	self.split_ix = {}
	self.iterators = {}
	for ix, sent in pairs(self.sents) do
		local split = sent['split']
		if not self.split_ix[split] then
			self.split_ix[split] = {}
			self.iterators[split] = 1
		end
		table.insert(self.split_ix[split], ix)
	end
end

function BoxLoader:getNumSents(opt)
	return #self.split_ix[opt.split]
end

function BoxLoader:getBoxes(sent)
	local image_id = sent['image_id']
	return self.imageToBoxes[tostring(image_id)]
end

function BoxLoader:newSent(opt)
	local split = utils.getopt(opt, 'split', 'test')

	-- fetch boxes
	local iterator = self.iterators[split]
	local sent = self.sents[self.split_ix[split][iterator]]
	local image_id = sent['image_id']
	self.image = self.Images[image_id]
	self.boxes = self.imageToBoxes[tostring(image_id)]

	-- fetch label
	local tokens = sent['tokens']
	self.label = torch.zeros(1, self.seq_length):long()
	for i, wd in ipairs(tokens) do
		if i > self.seq_length then break end
		if self.word_to_ix[wd] ~= nil then
			self.label[1][i] = self.word_to_ix[wd]
		else
			self.label[1][i] = self.word_to_ix['UNK']
		end
	end

	-- initialize box_iterator
	self.box_iterator = 1

	-- get next sent iterator
	local si = self.iterators[split]
	local si_next = si+1
	if si_next <= #self.split_ix[split] then
		self.iterators[split] = si_next
		ix = self.split_ix[split][si]  -- get ix of sents
		return self.sents[ix]
	else
		return nil
	end
end

function BoxLoader:getBatch(opt)
	local batch_size = utils.getopt(opt, 'batch_size', 10)
	local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
	local label_batch = torch.LongTensor(batch_size, self.seq_length)
	local wrapped = false
	local infos = {}
	for i=1, batch_size do

		-- fetch the bbox
		bbox = self.boxes[self.box_iterator]

		-- process the next iterator
		self.box_iterator = self.box_iterator + 1
		if self.box_iterator > #self.boxes then self.box_iterator = 1; wrapped = true end

		-- compute nx, ny, nw, nh
		local x, y, w, h = unpack(bbox)
		local width, height = self.image['width'], self.image['height']
		local nx, ny, nw, nh = math.floor(x/width*self.max_image_size)+1, math.floor(y/height*self.max_image_size)+1, 
		math.max(math.floor(w/width*self.max_image_size), 1), math.max(math.floor(h/height*self.max_image_size),1)
		
		-- fetch the image
		local h5_id = self.image['h5_id']
		local img = self.h5_file:read('/images'):partial({h5_id,h5_id},{1,self.num_channels},{1,self.max_image_size},{1,self.max_image_size})
		img_batch_raw[i] = img

		-- fetch the sequence labels
		label_batch[i] = self.label

		-- and record associated info as well
		local info_struct = {}
		info_struct.bbox = {nx, ny, nw, nh}
		table.insert(infos, info_struct)
	end

	local data = {}
	data.images = img_batch_raw
	data.image_id = self.image['image_id']
	data.labels = label_batch:transpose(1,2):contiguous()
	data.bounds = {it_pos_now = self.box_iterator, it_max = #self.boxes, wrapped = wrapped}
	data.infos = infos
	return data
end

-------------------------------------------------------------------------------
-- Compute losses of given batch of (input, seq)
-------------------------------------------------------------------------------
function computeLosses(input, seq)
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  local losses = torch.zeros(N)
  for b=1,N do -- iterate over batches
  	local n = 0
    local first_time = true
    for t=2,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)

      -- fetch the index of the next token in the sequence
      local target_index
      if t-1 > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t-1,b}] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        losses[b] = losses[b] - input[{ t,b,target_index }] -- log(p)
        n = n + 1
      end
    end
    -- normalize current sequence by number of effective words
    if n == 0 then losses[b] = 0 else losses[b] = losses[b]/n end
  end

  return losses
end

-------------------------------------------------------------------------------
-- visualization tool that requires torch's opencv
-------------------------------------------------------------------------------
local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'
function convertFormat(ori_im)
	local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(3,1,1):typeAs(ori_im)  -- in RGB order
	local im = ori_im:clone()
	im:add(1, vgg_mean:expandAs(im))
	im = im:index(1, torch.LongTensor{3,2,1})  -- convert to Opencv's BGR order.
	im = im:transpose(1,2):transpose(2,3)
	im = im:byte()
	return im
end
function visualize(gimgs, rimgs)
	for i = 1, gimgs:size(1) do
		gim, rim = gimgs[i], rimgs[i]
		gim = convertFormat(gim)
		rim = convertFormat(rim)
		local twoimgs = torch.zeros(224, 448, 3):byte()
		twoimgs[{ {}, {1, 224}, {} }] = gim
		twoimgs[{ {}, {225, 448}, {} }] = rim
		cv.imshow{'twoimgs', twoimgs}
		cv.waitKey{delay=0}
	end
end