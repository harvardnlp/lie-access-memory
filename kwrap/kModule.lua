require 'nn'
require 'nngraph'
require 'pl'
require 'paths'
paths.dofile('kWrap.lua')
local st = paths.dofile('split_transfer.lua')
local Om = OrderedMap

local kModule
if not nn.kModule then
    kModule = torch.class('nn.kModule', 'nn.kWrap')
else
    kModule = nn.kModule
end

function kModule:__init(inputs, outputs, allow_table_input)
    self.input_map = inputs
    self.output_map = outputs
    self.allow_table_input = allow_table_input
    ins = self:merge_inputs(inputs)
    outs = self:merge(outputs, outputs)
    self.mod = nn.gModule(ins, outs)
end