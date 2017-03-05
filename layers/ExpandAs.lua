require 'nn'
require 'nngraph'
local ExpandAs, parent
if not pcall(function()
            ExpandAs, parent = torch.class('nn.ExpandAs', 'nn.Module')
         end) then
    ExpandAs = nn.ExpandAs
    parent = nn.Module
end

-- ExpandAs(){A, B} returns A:expandAs(B)
function ExpandAs:__init(epsilon)
    parent.__init(self)
    self.gradInput = {}
end
function ExpandAs:updateOutput(inputs)
    self.output = inputs[1]:expandAs(inputs[2])
    return self.output
end
function ExpandAs:updateGradInput(inputs, gradOutput)
    self.gradInput = self.gradInput or {}
    self.gradInput[1] = self.gradInput[1] or inputs[1].new()
    self.gradInput[2] = self.gradInput[2] or inputs[2].new()
    self.gradInput[2]:resizeAs(inputs[2]):zero()
    self.temp = self.temp or gradOutput.new()
    self.temp:resizeAs(gradOutput):copy(gradOutput)
    local g = self.temp
    for d=1,gradOutput:dim() do
        if inputs[1]:size(d) == 1 then
            g:narrow(d, 1, 1):copy(g:sum(d))
            g = g:narrow(d, 1, 1)
        end
    end
    self.gradInput[1] = g
    return self.gradInput
end