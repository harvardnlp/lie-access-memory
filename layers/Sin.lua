require 'nn'
require 'totem'
local Sin, parent
-- componentwise cosine
if not pcall(function()
            Sin, parent = torch.class('nn.Sin', 'nn.Module')
         end) then
    Sin = nn.Sin
    parent = nn.Module
end

function Sin:updateOutput(in_)
    self.output = self.output or in_.new()
    self.output:resizeAs(in_):copy(torch.sin(in_))
    return self.output
end

function Sin:updateGradInput(in_, gradOut)
    self.gradInput = self.gradInput or in_.new()
    self.gradInput:resizeAs(in_):copy(torch.cmul(gradOut, torch.cos(in_)))
    return self.gradInput
end
