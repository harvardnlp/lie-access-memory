require 'nn'
require 'nngraph'
require 'pl'
require 'paths'
local st = paths.dofile('split_transfer.lua')
local Om = OrderedMap

local kWrap
if nn.kWrap then
    kWrap = nn.kWrap
else
    kWrap = torch.class('nn.kWrap', 'nn.Module')
end

function kWrap:__init(mod, input_map, output_map, allow_table_input)
    --[[
    @input_map: an Om, possibly layered (Oms)
    @output_map: an Om, possibly layered

    kWrap:updateOutput(input) will flatten input in order and feed it into
    mod:updateOutput if input has type Om. If input has type Map, then

    ]]
    self.mod = mod
    self.input_map = input_map
    self.output_map = output_map
    self.allow_table_input = allow_table_input
end

function kWrap:merge(input_map, in_)
    local input = List()
    local depth_first
    depth_first = function (kwmap, in_map)
        assert(not types.is_type(kwmap, Map), 'use either Om or List')
        if types.is_type(kwmap, Om) then
            for k, v in kwmap:iter() do
                assert(in_map,
                    'in_map is nil; possibly the original input map ' ..
                    'does not have all required keys')
                depth_first(v, in_map[k])
            end
        elseif types.is_type(kwmap, List) then
            for k, v in ipairs(kwmap) do
                assert(in_map,
                    'in_map is nil; possibly the original input map ' ..
                    'does not have all required keys')
                -- print('list', k, pp_om(in_map))
                depth_first(v, in_map[k])
            end
        else
            local maptype
            if not pcall(function() maptype = types.type(kwmap) end) then
                maptype = torch.type(kwmap)
            end
            if maptype == 'table' then
                print('warning:', kwmap,
                    'is a table; use Om or List instead if it is a table of keys')
            end
            input:append(in_map)
        end
    end
    depth_first(input_map, in_)
    return input
end

function kWrap:merge_inputs(in_)
    return self:merge(self.input_map, in_)
end

function kWrap:collect(output_map, out)
    local depth_first
    depth_first = function(kwmap, idx)
        assert(not types.is_type(kwmap, Map), 'use either Om or List')
        if types.is_type(kwmap, Om) then
            local m = Om()
            for k, v in kwmap:iter() do
                m[k] = depth_first(v, idx)
            end
            return m
        elseif types.is_type(kwmap, List) then
            local l = List()
            for _, v in ipairs(kwmap) do
                l:append(depth_first(v, idx))
            end
            return l
        else
            local maptype
            if not pcall(function() maptype = types.type(kwmap) end) then
                maptype = torch.type(kwmap)
            end
            if maptype == 'table' then
                print('warning:', kwmap, 'is a table; use Om or List instead if it is a table of keys')
            end
            idx[1] = idx[1] + 1
            return out[idx[1] - 1]
        end
    end
    it = {1}
    local outputs = depth_first(output_map, it)
    if it[1] <= #out then
        error('Output uncollected. Remaining: ' .. tablex.sub(out, it[1], #out):__tostring())
    end
    return outputs
end

function kWrap:collect_outputs(out)
    return self:collect(self.output_map, out)
end

function kWrap:submap(map1, map2, compare)
    local depth_first
    depth_first = function(m1, m2)
        if types.is_type(m1, Om) then
            if not types.is_type(m2, 'table') then
                return false
            end

            for k, v in m1:iter() do
                local b = depth_first(v, m2[k])
                if not b then return false end
            end
        elseif types.is_type(m1, List) then
            if not types.is_type(m2, 'table') then
                return false
            end
            for k, v in ipairs(m1) do
                local b = depth_first(v, m2[k])
                if not b then return false end
            end
        else
            if compare=='keys' then
                return not (m1 == nil and m2 ~= nil or m1 ~= nil and m2 == nil)
            else
                return compare(m1, m2)
            end
        end
        return true
    end
    return depth_first(map1, map2)
end

function kWrap:mapeq(a, b, mode)
    return self:submap(a, b, mode) and self:submap(b, a, mode)
end

function kWrap:forward(in_)
    assert(self.allow_table_input or
        torch.isTensor(in_) or
        types.is_type(in_, Om) or
        types.is_type(in_, List)
    )
    if not torch.isTensor(in_) then
        if not types.is_type(in_, List) then
            in_ = self:merge(self.input_map, in_)
        end
        if types.is_type(in_, 'table') and #in_ == 1 then
            in_ = in_[1]
        end
    end
    local out = self.mod:forward(in_)
    if not types.is_type(out, 'table') then
        out = {out}
    end

    self.output = self:collect(self.output_map, out)
    return self.output
end

function kWrap:updateOutput(in_)
    return self:forward(in_)
end


function kWrap:backward(in_, gradOut)
    assert(self.allow_table_input or
        (torch.isTensor(in_) or
        types.is_type(in_, Om) or
        types.is_type(in_, List)) and
        (types.is_type(gradOut, Om) or
        types.is_type(gradOut, List))
    )
    if not torch.isTensor(in_) then
        if not types.is_type(in_, List) then
            in_ = self:merge(self.input_map, in_)
        end
        if types.is_type(in_, 'table') and #in_ == 1 then
            in_ = in_[1]
        end
    end
    if not types.is_type(gradOut, List) then
        gradOut = self:merge(self.output_map, gradOut)
    end
    if types.is_type(gradOut, 'table') and #gradOut == 1 then
        gradOut = gradOut[1]
    end
    local gradIn = self.mod:backward(in_, gradOut)
    if not types.is_type(gradIn, 'table') then
        gradIn = {gradIn}
    end
    self.gradInput = self:collect(self.input_map, gradIn)
    return self.gradInput
end

function kWrap:updateGradInput(in_, gradOut)
    return self:backward(in_, gradOut)
end

------------------------------------
--[[wrappers for nn.Module methods]]
------------------------------------

function kWrap:clone(...)
    return nn.kWrap(self.mod:clone(...), self.input_map, self.output_map)
end

function kWrap:parameters()
    return self.mod:parameters()
end

function kWrap:getParameters()
    return self.mod:getParameters()
end

function kWrap:training()
    return self.mod:training()
end

function kWrap:evaluate()
    return self.mod:evaluate()
end

function kWrap:type(...)
    self.mod = self.mod:type(...)
    return self
end

function kWrap:float(...)
    self.mod = self.mod:float(...)
    return self
end

function kWrap:double(...)
    self.mod = self.mod:double(...)
    return self
end


function kWrap:cuda(...)
    self.mod = self.mod:cuda(...)
    return self
end

function kWrap:cl(...)
    self.mod = self.mod:cl(...)
    return self
end