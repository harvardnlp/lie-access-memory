require 'nn'

local InvNormalize, parent
if not pcall(function()
            InvNormalize, parent = torch.class('nn.InvNormalize', 'nn.Module')
         end) then
    InvNormalize = nn.InvNormalize
    parent = nn.Module
end
--[[
InvNormalize(p, eps)(t)
where t is a nonegative matrix of size (batch_size, mem_length)
computes 1/(t + eps)^p with each row L1 normalized
]]
function InvNormalize:__init(p, eps)
    parent.__init(self)
    self.p = p or 1
    self.eps = eps or 1e-10
end

function InvNormalize:updateOutput(input)
    --[[
    we compute
        row_min_raw = min(input, 2)
        row_min_over_raw = row_min_raw / input
        row_sum = sum(row_min_over_raw, 2)
        output = row_min_over_raw / row_sum
    ]]
    self.row_min_raw = self.row_min or input.new() -- a_min in notes
    self.row_min_over_raw = self.row_min_over_raw or input.new() -- v in notes
    self.row_sum = self.row_sum or input.new() -- U in notes

    -- row_min has size (batch_size, 1)
    self.row_min_raw:resize(input:size(1), 1):min(input, 2)
    self.row_min_raw:add(self.eps)
    -- row_min_over_raw has size (batch_size, mem_length)
    self.row_min_over_raw:resizeAs(input)
    self.row_min_over_raw:copy(self.row_min_raw:expandAs(input))
    input:add(self.eps)

    self.row_min_over_raw:cdiv(input)

    input:add(-self.eps)

    self.row_min_over_raw:pow(self.p)

    -- row_sum has size (batch_size, 1)
    self.row_sum:resize(input:size(1), 1):norm(self.row_min_over_raw, 1, 2)

    -- output has size (batch_size, mem_length)
    self.output:resizeAs(input):copy(self.row_min_over_raw)
    self.output:cdiv(self.row_sum:expandAs(input))

    self.backward_ready = true
    return self.output
end

function InvNormalize:updateGradInput(input, gradOutput)
    --[[
    We compute
        gradInput = (v.dot(H - H^T) / a_min) * v^2 / U^2
    ]]
    assert(self.backward_ready)
    self.gradInput = self.gradInput or input.new()
    self.w = self.w or input.new()

    local batch_size = input:size(1)
    local M = gradOutput:size(2)
    local g = gradOutput:view(batch_size, 1, M)
    self.w:resize(batch_size, 1, 1)
    self.w:bmm(g, self.output:view(batch_size, M, 1))
    self.w = self.w:view(batch_size, 1)

    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    self.gradInput:csub(self.w:expandAs(input))
    -- this destroys row_min_over_raw, but we can just retrieve it from input if necessary
    self.row_min_over_raw:pow((self.p+1)/self.p)
    self.gradInput:cmul(self.row_min_over_raw)
    self.gradInput:mul(-self.p)
    -- this destroys row_min_raw
    self.gradInput:cdiv(self.row_min_raw:cmul(self.row_sum):expandAs(input))

    self.backward_ready = false
    return self.gradInput
end