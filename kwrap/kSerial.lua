require 'nn'
require 'nngraph'
require 'pl'
require 'paths'
local utils = paths.dofile('../utils/nested.lua')
local Om = OrderedMap

local kSerial
if nn.kSerial then
    kSerial = nn.kSerial
else
    kSerial = torch.class('nn.kSerial')
end

function kSerial:__init(kwrap)
    --[[
    converts kwrap's input and output maps into tables for serialization via torch
    ]]
    if torch.isTypeOf(kwrap, nn.kWrap) or torch.isTypeOf(kwrap, nn.kModule) then
        self.__type__ = 'kStuff'
        self.mod = kwrap.mod
        self.input_map = utils.dump2table(kwrap.input_map)
        self.output_map = utils.dump2table(kwrap.output_map)
    elseif types.is_type(kwrap, Map) or types.is_type(kwrap, Om) or types.is_type(kwrap, List) then
        self.__type__ = 'nested_collection'
        self.dumped = utils.dump2table(kwrap, {keep_tensor=true})
    else
        error('no implementation for serializing ' .. tostring(kwrap))
    end
end

function kSerial:load()
    if self.__type__ == 'kStuff' then
        return nn.kWrap(self.mod, utils.loadfromtable(self.input_map), utils.loadfromtable(self.output_map))
    elseif self.__type__ == 'nested_collection' then
        return utils.loadfromtable(self.dumped)
    else
        error('unknown __type__', self.__type__)
    end
end