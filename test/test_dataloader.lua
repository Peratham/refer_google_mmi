package.path = '../?.lua;' .. package.path
require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
local utils = require 'misc.utils'
require 'misc.DataLoader'
local net_utils = require 'misc.net_utils'
require 'cudnn'
-- cv load
local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'

-------------------------------------------------------------------------------
-- Test data loader
-------------------------------------------------------------------------------
torch.manualSeed(24)
local loader = DataLoader{h5_file = '../data/refcoco_licheng/data.h5', json_file = '../data/refcoco_licheng/data.json'}
loader:shuffle('train')

-- transfer loaded data to opencv's format
function convertFormat(ori_im)
	local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(3,1,1):typeAs(ori_im)  -- in RGB order
	local im = ori_im:clone()
	im:add(1, vgg_mean:expandAs(im))
	im = im:index(1, torch.LongTensor{3,2,1})  -- convert to Opencv's BGR order.
	im = im:transpose(1,2):transpose(2,3)
	im = im:byte()
	return im
end

-- -- look at one validation batch
-- local data = loader:getBatch{batch_size = 30, split = 'val', seq_per_img = 3}
-- local ix_to_word = loader:getVocab()
-- local gimgs, rimgs, lfeats = net_utils.prepro(data.images, data.infos, 'bbox', true, 1)
-- local gim, rim, loc
-- for i = 1, gimgs:size(1) do
-- 	gim = gimgs[i]
-- 	rim = rimgs[i]
-- 	loc = lfeats[i]
-- 	--  convert format
-- 	gim = convertFormat(gim)
-- 	rim = convertFormat(rim)
-- 	-- combine the global and region images 
-- 	local twoimgs = torch.zeros(224, 448, 3):byte()
-- 	twoimgs[{ {}, {1, 224}, {} }] = gim
-- 	twoimgs[{ {}, {225, 448}, {} }] = rim
-- 	-- show location
-- 	print(string.format('tl=(%.2f, %.2f), br=(%.2f, %.2f), area=%.2f', loc[1], loc[2], loc[3], loc[4], loc[5]))
-- 	captions = data.labels[{ {}, {(i-1)*3+1, i*3} }] 
-- 	-- show sent
-- 	for j = 1, 3 do
-- 		local sent = ''
-- 		for t = 1, 10 do
-- 			if captions[t][j] ~= 0 then
-- 				w = ix_to_word[tostring(captions[t][j])]
-- 				sent = sent .. ' ' .. w
-- 			end
-- 		end
-- 		print(sent)
-- 	end
-- 	cv.imshow {'twoimgs', twoimgs}
-- 	cv.waitKey {delay=0}
-- end

-- look at one training batch
local data = loader:getPosNegBatch{batch_size = 5, split = 'train', seq_per_img = 3}
local ix_to_word = loader:getVocab()
local gimgs1, rimgs1, lfeats1 = net_utils.prepro(data.images, data.infos, 'pos_bbox', true, 1)
local gimgs2, rimgs2, lfeats2 = net_utils.prepro(data.images, data.infos, 'neg_bbox', true, 1)

local gim1, rim1, loc1
local gim2, rim2, loc2
for i = 1, gimgs1:size(1) do
	gim1, rim1, loc1 = gimgs1[i], rimgs1[i], lfeats1[i]
	gim2, rim2, loc2 = gimgs2[i], rimgs2[i], lfeats2[i]

	--  convert format
	gim1 = convertFormat(gim1)
	rim1 = convertFormat(rim1)
	gim2 = convertFormat(gim2)
	rim2 = convertFormat(rim2)

	-- combine the global and region images 
	local twoimgs = torch.zeros(448, 448, 3):byte()
	twoimgs[{ {1, 224}, {1, 224}, {} }] = gim1
	twoimgs[{ {1, 224}, {225, 448}, {} }] = rim1
	twoimgs[{ {225, 448}, {1, 224}, {} }] = gim2
	twoimgs[{ {225, 448}, {225, 448}, {} }] = rim2

	-- show location
	print(string.format('pos_bbox: tl=(%.2f, %.2f), br=(%.2f, %.2f), area=%.2f', loc1[1], loc1[2], loc1[3], loc1[4], loc1[5]))
	print(string.format('neg_bbox: tl=(%.2f, %.2f), br=(%.2f, %.2f), area=%.2f', loc2[1], loc2[2], loc2[3], loc2[4], loc2[5]))
	captions = data.labels[{ {}, {(i-1)*3+1, i*3} }] 
	-- show sent
	for j = 1, 3 do
		local sent = ''
		for t = 1, 10 do
			if captions[t][j] ~= 0 then
				w = ix_to_word[tostring(captions[t][j])]
				sent = sent .. ' ' .. w
			end
		end
		print(sent)
	end
	cv.imshow {'twoimgs', twoimgs}
	cv.waitKey {delay=0}
end

