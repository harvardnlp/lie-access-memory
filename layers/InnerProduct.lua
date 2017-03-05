require 'nn'
require 'totem'
local InnerProduct, parent
if not pcall(function()
            InnerProduct, parent = torch.class('nn.InnerProduct', 'nn.Module')
         end) then
    InnerProduct = nn.InnerProduct
    parent = nn.Module
end

-- InnerProduct()(A, BList) takes a (batched) vector A and a list of (batched) vectors Blist
-- and compute the inner product of A with each element of BList, and return this as a tensor
-- if A has size (batch, M), and BList is a list of length L of the same sized tensor, then
-- return a tensor of size (batch, L)
function InnerProduct:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), {}}
end

function InnerProduct:updateOutput(input)
    local A, BList = input[1], input[2]
    assert(#BList > 0, 'second input must be a nonempty list')
    -- assuming A has size (batch, M) and BList is a list of tensors of the same size
    local batch_size = A:size(1)
    local M = A:size(2)
    local _A = A:view(batch_size, 1, M)
    if not self.output then
        self.output = A.new()
    end
    self.output:resize(batch_size, #BList, 1)
    for i = 1, #BList do
        local _B = BList[i]:view(batch_size, M, 1)
        self.output:narrow(2, i, 1):bmm(_A, _B)
    end
    self.output:view(self.output, batch_size, #BList)
    assert(self.output:size(1) == batch_size)
    assert(self.output:size(2) == #BList)
    return self.output
end

function InnerProduct:updateGradInput(input, gradOutput)
    local A, BList = unpack(input)

    if #self.gradInput ~= 2 then
      local gradA = self.gradInput[1] or A.new()
      self.gradInput = {gradA, {}}
    end
    self._gradInput2 = self._gradInput2 or {}
    for i = 1, #BList do
        self._gradInput2[i] = self._gradInput2[i] or BList[i].new()
        self._gradInput2[i]:resizeAs(BList[i])
    end
    self.gradInput[2] = {}
    self.gradInput[1]:resizeAs(A):zero()
    for i = 1, #BList do
        self.gradInput[2][i] = self._gradInput2[i] or A.new()
        local g = gradOutput[{{}, {i, i}}]:expandAs(A)
        self.gradInput[1]:addcmul(BList[i], g)
        self.gradInput[2][i]:resizeAs(A):cmul(A, g)
    end
    return self.gradInput
end
