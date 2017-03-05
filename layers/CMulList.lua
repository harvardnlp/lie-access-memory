require 'nn'
local CMulList, parent
if not pcall(function()
                CMulList, parent = torch.class('nn.CMulList', 'nn.CMulTable')
            end) then
     CMulList = nn.CMulList
     parent = nn.CMulTable
end

function CMulList:updateOutput(input)
    if torch.isTensor(input[1]) then
        return parent.updateOutput(self, input)
    end
    -- if tensor has size (batchsize, L), then
    -- list should be a list of L tensors each of size (batchsize, 1)
    list, tensor = unpack(input)
    self.output = self.output or tensor.new()
    self.output:resizeAs(tensor)
    for i = 1, #list do
        -- the conditional below is needed to circumvent a cmul bug
        -- see https://github.com/torch/torch7/issues/674
        local o
        if list[i]:dim() == 2 then
            o = self.output[{{}, {i, i}}]
        elseif list[i]:dim() == 1 then
            o = self.output[{{}, i}]
        else
            error('list[' .. i .. ']:dim() == ' .. list[i]:dim())
        end
        o:cmul(list[i], tensor[{{}, i}])
    end
    return self.output
end

function CMulList:updateGradInput(input, gradOutput)
    if torch.isTensor(input[1]) then
        return parent.updateGradInput(self, input, gradOutput)
    end
    -- if tensor has size (batchsize, L), then
    -- list should be a list of L tensors each of ize (batchsize, 1)
    list, tensor = unpack(input)
    self.gradList = {}
    self.gradTensor = self.gradTensor or tensor.new()
    self.gradTensor:resizeAs(tensor)
    for i = 1, #list do
        local gt
        if list[i]:dim() == 2 then
            gt = self.gradTensor[{{}, {i, i}}]
        elseif list[i]:dim() == 1 then
            gt = self.gradTensor[{{}, i}]
        else
            error('list[' .. i .. ']:dim() == ' .. list[i]:dim())
        end
        gt:cmul(list[i], gradOutput[{{}, i}])

        self.gradList[i] = self.gradList[i] or tensor.new()
        self.gradList[i]:resizeAs(tensor[{{}, i}])
        self.gradList[i]:cmul(tensor[{{}, i}], gradOutput[{{}, i}])
    end
    self.gradInput = {self.gradList, self.gradTensor}
    return self.gradInput
end