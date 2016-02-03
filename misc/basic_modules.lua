require 'nn'

local basic_utils = {}

-------------------------------------------------------------------------------
-- Basic Module - nn.Scale
-------------------------------------------------------------------------------
-- layer that scale the input so that we can add weights on nn.Module.
-- it's like nn.Identity()*const, but I haven't found such scaling module in module.
local scale, parent = torch.class('nn.Scale', 'nn.Module')
function scale:__init(s)
  parent.__init(self)
  self.s = s
end
function scale:updateOutput(input)
  if self.s == 1 then self.output = input; return self.output end
  self.output:resizeAs(input):copy(input)
  self.output:mul(self.s)
  return self.output
end
function scale:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  if self.s ~= 0 then
    self.gradInput:mul(self.s)
  else
    self.gradInput:zero()
  end
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Basic Module - nn.FeatExpander
-------------------------------------------------------------------------------
-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model - ExtractPos module
-- output is fed to the language model loss
-------------------------------------------------------------------------------
local layer, parent = torch.class('nn.ExtractPositive', 'nn.Module')
function layer:__init()
  parent.__init(self)
end
--[[
Input is a Tensor of size (D+2)x(N*2)x(M+1)
We want to forward the first half (D+2)xNx(M+1)
--]]
function layer:updateOutput(input)
  local L, NpN, Mp1 = input:size(1), input:size(2), input:size(3)  -- Note here, NpN (N plus N) = 2*batch_size
  local N = NpN/2
  assert(N*2 == NpN)
  self.output = input[{{}, {1,N}, {}}]
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  local N = input:size(2)/2
  self.gradInput:resizeAs(input):zero()
  self.gradInput[{{}, {1,N}, {}}] = gradOutput
  return self.gradInput
end
-------------------------------------------------------------------------------
-- Language Model - SentLogProb module
-- output is fed to the marginal ranking loss
-------------------------------------------------------------------------------
local layer, parent = torch.class('nn.SentLogProb', 'nn.Module')
function layer:__init()
  parent.__init(self)
end

--[[
Input is a Table of {Tensor of size (D+2)x(N*2)x(M+1), Tensor of size DxN}
- The first Tensor is [ pos | neg ]
-  the 1st half (D+2)xNx(M+1) are the logprobs of positive pairs (Img_pos, Sent_pos)
-  the 2nd half (D+2)xNx(M+1) are the logprobs of negative pairs (Img_neg, Sent_pos)
- The second Tensor is seq of size DxN
Ouput is a table of {Tensor of size N, Tensor of size N}
- The first tensor is the score of N pairs ((Img_pos, Sent_pos)
- The Second tensor is the score of N pairs ((Img_neg, Sent_pos)
--]]
function layer:updateOutput(input)
  local logProbs, seq = input[1], input[2]
  local L, NpN, Mp1 = logProbs:size(1), logProbs:size(2), logProbs:size(3)  -- Note here, NpN (N plus N) = 2*batch_size
  local D, N = seq:size(1), seq:size(2)
  assert(D == L-2, 'input Tensor should be 2 larger in time')
  assert(NpN == N*2, 'input batch_size should be 2 times of seq\'s')

  -- Initialize
  self.output = {torch.zeros(N), torch.zeros(N)}  -- first table is the positive pair for N (img, sent)s, second is negative pair

  local loss = 0
  for b = 1,N do
    local first_time = true
    local n = 0
    for t=2, L do
      -- fetch the index of the next token in the sequence
      local target_index
      if t-1 > D then
        target_index = 0
      else
        target_index = seq[{t-1, b}]
      end

      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      if target_index ~= 0 then
        n = n + 1
        -- b-th positive pair
        self.output[1][b] = self.output[1][b] + logProbs[{t, b, target_index}]  
        -- b-th negative pair
        self.output[2][b] = self.output[2][b] + logProbs[{t, N+b, target_index}]
      end
    end
    self.output[1][b] = self.output[1][b] / n
    self.output[2][b] = self.output[2][b] / n
  end
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local logProbs, seq = input[1], input[2]
  local L, NpN, Mp1 = logProbs:size(1), logProbs:size(2), logProbs:size(3)  -- Note here, NpN (N plus N) = 2*batch_size
  local D, N = seq:size(1), seq:size(2)

  local pos_grad = gradOutput[1]  -- of size N
  local neg_grad = gradOutput[2]  -- of size N
  local gradLogProbs = torch.zeros(logProbs:size())  -- reset to zeros, size : L x 2N x Mp1

  for b = 1,N do
    local first_time = true
    local n = 0
    for t=2, L do
      -- fetch the index of the next token in the sequence
      local target_index
      if t-1 > D then
        target_index = 0
      else
        target_index = seq[{t-1, b}]
      end

      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      if target_index ~= 0 then
        n = n + 1
        -- b-th positive pair
        gradLogProbs[{t, b, target_index}] = pos_grad[b]
        -- b-th negative pair
        gradLogProbs[{t, N+b, target_index}] = neg_grad[b]
      end
    end

    gradLogProbs[{t, b}] = gradLogProbs[{t, b}]:div(n)
    gradLogProbs[{t, N+b}] = gradLogProbs[{t, N+b}]:div(n)
  end

  self.gradInput = {gradLogProbs, torch.Tensor()}
  return self.gradInput
end


-------------------------------------------------------------------------------
-- Important layer feeding to crit1 and crit2
-------------------------------------------------------------------------------
function basic_utils.FeedToCrits(ranking_weight)
  local feed = nn.ConcatTable()
  -- output (LxNxMp1) to be fed to crit1
  feed:add(nn.ExtractPositive())
  -- output table of (L x 2N x Mp1) to be fed to SentLogProb and crit2
  feed:add(nn.Scale(ranking_weight))
  -- return
  return feed
end

return basic_utils