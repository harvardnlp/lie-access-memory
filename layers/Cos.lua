require 'nn'
require 'totem'
local Cos, parent
-- componentwise cosine
if not pcall(function()
            Cos, parent = torch.class('nn.Cos', 'nn.Module')
         end) then
    Cos = nn.Cos
    parent = nn.Module
end

function Cos:updateOutput(in_)
    self.output = self.output or in_.new()
    self.output:resizeAs(in_):copy(torch.cos(in_))
    return self.output
end

function Cos:updateGradInput(in_, gradOut)
    self.gradInput = self.gradInput or in_.new()
    self.gradInput:resizeAs(in_):copy(torch.cmul(gradOut, -torch.sin(in_)))
    return self.gradInput
end
