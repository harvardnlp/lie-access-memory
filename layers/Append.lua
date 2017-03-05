require 'nn'
require 'nngraph'
require 'totem'
local Append, parent
if not pcall(function()
                Append, parent = torch.class('nn.Append', 'nn.Module')
            end) then
     Append = nn.Append
     parent = nn.Module
end
function Append:__init()
    parent.__init(self)
    self.output = List()
    self.gradInput = {}
end

function Append:updateOutput(input)
    oldParts, newPart, parentNodes = unpack(input)
    self.output = List(tablex.copy(oldParts))
    self.oldSize = #self.output
    table.insert(self.output, newPart)
    self.newSize = #self.output
    return self.output
end

function Append:updateGradInput(input, gradOutput)
    local list, newPart, parentNodes = unpack(input)

    self.gradInput = {}
    assert(#gradOutput == self.newSize)
    assert(#list == self.oldSize)
    self.gradInput[1] = tablex.sub(gradOutput, 1, self.oldSize)
    self.gradInput[2] = gradOutput[self.newSize]

    self.gradParents = self.gradParents or {}
    if type(parentNodes) ~= 'table' then
        self.gradParents[1] = self.gradParents[1] or newPart.new()
        self.gradParents[1]:resizeAs(parentNodes):zero()
        self.gradInput[3] = self.gradParents[1]
    else
        for i, node in ipairs(parentNodes) do
            self.gradParents[i] = self.gradParents[i] or newPart.new()
            self.gradParents[i]:resizeAs(node):zero()
        end
        self.gradInput[3] = self.gradParents
    end
    return self.gradInput
end

function Append:type(type, tensorCache)
    self.gradInput = {}
    for i = 1, #self.output do
        self.output[i]:type(type, tensorCache)
    end
end