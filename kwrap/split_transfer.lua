require 'pl'
require 'nn'
require 'nngraph'

local Om = OrderedMap
local st = {}
function st.get_split_sizes(transfer_tbl)
    local split_sizes = OrderedMap()
    for k, v in transfer_tbl:iter() do
        split_sizes[k] = st.get_total_size(v)
    end
    return split_sizes
end
function st.get_total_size(transfer_tbl)
    local flattened = st.flatten_values(transfer_tbl)
    return tablex.reduce('+', flattened)
end
function st.split_transfer(input, dim, transfer_tbl)
    local _split
    local out_tbl = OrderedMap()
    local split_sizes = st.get_split_sizes(transfer_tbl)
    local transfer_splits = OrderedMap()
    local idx = 1
    for transfer, length in split_sizes:iter() do
        transfer_splits[transfer] = transfer(nn.Narrow(dim, idx, length)(input))
        idx = idx + length
    end

    local function _split(tbl, in_, i)
        if type(tbl) == 'number' then
            return nn.Narrow(dim, i, tbl)(in_), i + tbl
        elseif types.is_type(tbl, List) then
            local t = List()
            for k, v in ipairs(tbl) do
                t[k], i = _split(v, in_, i)
            end
            return t, i
        elseif types.is_type(tbl, OrderedMap) or types.is_type(tbl, Map) then
            local t = OrderedMap()
            for k, v in tbl:iter() do
                t[k], i = _split(v, in_, i)
            end
            return t, i
        else
            error('unknown type: ' .. pretty.write(tbl))
        end
    end

    for transfer, subtbl in transfer_tbl:iter() do
        out_tbl[transfer] = _split(subtbl, transfer_splits[transfer], 1)
    end
    return out_tbl
end

function st.flatten_values(arr)
    local result = List()

    local function flatten(arr)
        if types.is_type(arr, Map) or types.is_type(arr, OrderedMap) then
            for _, v in arr:iter() do
                if types.is_type(v, "table") and not torch.isTypeOf(v, 'nngraph.Node') then
                    flatten(v)
                else
                    table.insert(result, v)
                end
            end
        elseif types.is_type(arr, List) then
            for v in arr:iter() do
                if types.is_type(v, "table") and not torch.isTypeOf(v, 'nngraph.Node') then
                    flatten(v)
                else
                    table.insert(result, v)
                end
            end
        else
            error('unimplemented arr type ' .. types.type(arr))
        end
    end

    flatten(arr)
    return result
end

return st
