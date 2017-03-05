require 'nn'
local PowSum, parent
if not pcall(function()
            PowSum, parent = torch.class('nn.PowSum', 'nn.Module')
         end) then
    PowSum = nn.PowSum
    parent = nn.Module
end

-- PowSum(p, dim)(A) computes the p-powered sum of A in dimension dim
function PowSum:__init(p, dim)
    parent.__init(self)
    self.dim = dim
    self.p = p
end

function PowSum:updateOutput(input)
    self.output = torch.norm(input, self.p, self.dim):squeeze(self.dim)
    self.output:pow(self.p)
    return self.output
end

function PowSum:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInput or input.new()
    self.gradInput:resizeAs(input):copy(input):pow(self.p-1):mul(self.p)
    local size = input:size():totable()
    size[self.dim] = 1
    local _gradOut = gradOutput:view(unpack(size))
    self.gradInput:cmul(_gradOut:expandAs(self.gradInput))
    return self.gradInput
end