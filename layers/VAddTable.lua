require 'nn'
local VAddTable, parent
if not pcall(function()
            VAddTable, parent = torch.class('nn.VAddTable', 'nn.Module')
         end) then
    VAddTable = nn.VAddTable
    parent = nn.Module
end

-- VAddTable()(A, BList) takes a (batched) vector A and a list of (batched) vectors Blist
-- and compute the sum of A with each element of BList, and return this as a tensor
-- if A has size (batch, M), and BList is a list of length L of the same sized tensor, then
-- return a tensor of size (batch, L, M)
function VAddTable:__init(neg_A)
    -- if neg_A, then compute B - A instead of B + A
    parent.__init(self)
    self.gradInput = {torch.Tensor(), {}}
    self.neg_A = neg_A
end

function VAddTable:updateOutput(input)
    local A, BList = input[1], input[2]
    assert(#BList > 0, 'second input must be a nonempty list')
    -- assuming A has size (batch, M) and BList is a list of tensors of the same size
    local batch_size = A:size(1)
    local M = A:size(2)
    local _A = A:view(batch_size, 1, M)
    if not self.output then
        self.output = A.new()
    end
    self.output:resize(batch_size, #BList, M)
    for i = 1, #BList do
        local _B = BList[i]:view(batch_size, 1, M)
        if self.neg_A then
            self.output:narrow(2, i, 1):csub(_B, _A)
        else
            self.output:narrow(2, i, 1):add(_A, _B)
        end
    end
    assert(self.output:size(1) == batch_size)
    assert(self.output:size(2) == #BList)
    assert(self.output:size(3) == M)
    return self.output
end

function VAddTable:updateGradInput(input, gradOutput)
    local A, BList = unpack(input)
    local batch_size = A:size(1)
    local M = A:size(2)

    if #self.gradInput ~= 2 then
      local gradA = self.gradInput[1] or A.new()
      self.gradInput = {gradA, {}}
    end
    self.gradInput[2] = {}
    self.gradInput[1]:resizeAs(A):zero()
    for i = 1, #BList do
        local g = gradOutput[{{}, i, {}}]
        if self.neg_A then
            self.gradInput[1]:csub(g)
        else
            self.gradInput[1]:add(g)
        end
        self.gradInput[2][i] = g
    end
    return self.gradInput
end
