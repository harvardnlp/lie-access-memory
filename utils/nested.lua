require 'pl'
require('pl.stringx').import()
local common = paths.dofile('common.lua')
local Om = OrderedMap
local nested = {}
function nested.stateslice(rnn_states, key_cascade, range, kw)
    -- key_cascade is a list of keys (or indices of lists or tensors)
    -- for example, {'debug', 'memory', 1, 'gate_read', 1}
    kw = kw or {}
    totable = kw.totable == nil and true or kw.totable
    set = kw.set
    if type(range) == 'table' and #range == 2 then
        range = torch.range(unpack(range))
    else
        range = range or torch.range(1, #rnn_states)
    end
    local gates = tablex.imap(
        function (i)
            rnn_states[i] = rnn_states[i] or Om()
            local s = rnn_states[i]
            if set then
                for j, key in ipairs(key_cascade) do
                    if j == #key_cascade then
                        break
                    end
                    s[key] = s[key] or Om()
                    s = s[key]
                end
                s[key_cascade[#key_cascade]] = set[i]
                return set[i]
            else
                for _, key in ipairs(key_cascade) do
                    s = s[key]
                end
                if totable and torch.isTensor(s) then
                    return torch.totable(s:double())
                else
                    return s
                end
            end
        end,
        type(range) == 'table' and range or torch.totable(range))
    return range, gates
end

function nested.sslice(...)
    local _, a = nested.stateslice(...)
    return a
end

function nested.mapprod(name_list)
    -- mapprod(Om{
    --         {a={1, 2, 3}},
    --         {b={9, 8, 7}},
    --     }) is
    -- {{a=1,b=9},{a=1,b=8},{a=1,b=7},{a=2,b=9},{a=2,b=8},{a=2,b=7},{a=3,b=9},{a=3,b=8},{a=3,b=7}}
    local res = List()
    local function _prod(current, remain)
        if #remain == 0 then
            res:append(OrderedMap(current))
        else
            local name = remain:pop(1)
            for i, v in ipairs(name_list[name]) do
                current:set(name, v)
                _prod(current, remain)
            end
            current:set(name, nil)
            remain:insert(1, name)
        end
    end
    _prod(OrderedMap(), name_list:keys())
    return res
end
function nested.submap(map1, map2, compare)
    -- compares nested Om/List as map1 and any nested table as map2
    local depth_first
    depth_first = function(m1, m2)
        if types.is_type(m1, Om) or types.is_type(m1, Map) then
--             print('Om')
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

function nested.mapeq(a, b, compare, assertorder)
    -- compares nested Om/List/Map; results for other nested tables are arbitrary
    if not assertorder then
        return nested.submap(a, b, compare) and nested.submap(b, a, compare)
    else
        local function depth_first(m1, m2)
            if types.is_type(m1, Om) or types.is_type(m1, Map) then
                if not types.is_type(m2, Om) and not types.is_type(m2, Map) or m1:keys() ~= m2:keys() then
                    return false
                end

                for k, v in m1:iter() do
                    local b = depth_first(v, m2[k])
                    if not b then return false end
                end
            elseif types.is_type(m1, List) then
                if not types.is_type(m2, 'table') or #m1 ~= #m2 then
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
        return depth_first(a, b)
    end
end
function nested.nestmap2(a, b, fun, kwargs)
    kwargs = kwargs or {}
    local function depth_first(m1, m2)
        if m2 == nil then
            return
        end
        if types.is_type(m1, Om) or types.is_type(m1, Map) then
            for k, v in m1:iter() do
                depth_first(v, m2[k])
            end
        elseif types.is_type(m1, List) then
            for k, v in ipairs(m1) do
                depth_first(v, m2[k])
            end
        else
            fun(m1, m2)
        end
        return true
    end
    depth_first(a, b)
    return a
end

function nested.nestmap(a, fun, kwargs)
    kwargs = kwargs or {}
    local function depth_first(m1)
        if types.is_type(m1, Om) or types.is_type(m1, Map) then
            for k, v in m1:iter() do
                depth_first(v)
            end
        elseif types.is_type(m1, List) then
            for k, v in ipairs(m1) do
                depth_first(v)
            end
        else
            fun(m1)
        end
        return true
    end
    depth_first(a)
    return a
end

function nested.nestadd(a, b)
    return nested.nestmap2(a, b, function(x, y) return x:add(y) end)
end

function nested.zero_nested(map, opt, forcezero, template)
    -- template is a map of zero tensors such that the returned tensors are narrowed versions of the templates
    -- ignore all keys of type '__abcde__'
    local function df(m, tpl)
        if types.is_type(m, Om) or types.is_type(m, Map) then
            local zeroed = Om()
            for k, v in m:iter() do
                if not (type(k) == 'string' and k:startswith('__') and k:endswith('__')) then
                    -- print(k)
                    zeroed[k] = df(v, tpl and tpl[k] or nil)
                end
            end
            return zeroed
        elseif types.is_type(m, List) then
            local zeroed = List()
            for i, v in ipairs(m) do
                zeroed[i] = df(v, tpl and tpl[i] or nil)
            end
            return zeroed
        elseif torch.isTensor(m) then
            if tpl and torch.isTensor(tpl) then
                -- assuming that we only need to narrow the second dimension; most relevant to memory
                if tpl:size(2) < m:size(2) then
                    tpl:resizeAs(m):zero()
                end
                return tpl:narrow(2, 1, m:size(2)):zero()
            elseif type(opt) == 'table' then
                return common.gpuzeros(opt)(m:size())
            elseif type(opt) == 'string' then
                return torch.Tensor(m:size()):type(opt)
            end
        else
            if forcezero then
                return 0
            elseif m ~= nil then
                error('unexpected type')
            end
        end
    end
    return df(map, template)
end
function nested.copy(from, mode, to, kwargs)
    -- mode can be either 'shallow' or 'deep', default begins 'shallow'
    -- if _to_ == nil and mode == 'deep', then return a fresh deep copy of from
    -- if _to_ is any nested Map/Om/List, then for keys of _from_ already in _to_, we retain the tensor of _to_ but copy the elements of _from_
    --   for keys of _from_ not in _to_, a new key is added, and the new copy of the tensor of _from_ is added.
    mode = mode or 'shallow'
    kwargs = kwargs or {}
    defaultval = kwargs.defaultval
    function df(m, tpl)
        if torch.isTensor(m) then
            if mode == 'deep' then
                if tpl then
                    if torch.isTensor(tpl) then
                        tpl:type(m:type()):resizeAs(m):copy(m)
                        return tpl
                    else
                        error('m is a tensor but tpl is not')
                    end
                else
                    return torch.Tensor(m:size()):type(m:type()):copy(m)
                end
            elseif mode == 'shallow' then
                if defaultval then
                    return type(defaultval) == 'function' and defaultval() or defaultval
                else
                    return m
                end
            else
                error('unknown mode')
            end
        elseif types.is_type(m, Om) or types.is_type(m, Map) then
            if tpl then
                for k, v in m:iter() do
                    if tpl[k] then
                        tpl[k] = df(v, tpl[k])
                    else
                        tpl[k] = df(v, nil)
                    end
                end
                return tpl
            else
                local t = Om()
                for k, v in m:iter() do
                    t[k] = df(v, nil)
                end
                return t
            end
        elseif types.is_type(m, List) then
            if tpl then
                for i, v in ipairs(m) do
                    if tpl[i] then
                        tpl[i] = df(v, tpl[i])
                    else
                        tpl[i] = df(v, nil)
                    end
                end
                return tpl
            else
                local t = List()
                for i, v in ipairs(m) do
                    t[i] = df(v, nil)
                end
                return t
            end
        else
            error('unexpected type from object ' .. tostring(m))
        end
    end
    return df(from, to)
end

function nested.deepcopy(from, to)
    return nested.copy(from, 'deep', to)
end

function nested.dump2table(map, kwargs)
    kwargs = kwargs or {}
    kwargs.keep_scalar = kwargs.keep_scalar or true
    local function dump(m)
        if types.is_type(m, Om) or types.is_type(m, Map) then
            local d = {'__Om__'}
            for k, v in m:iter() do
                table.insert(d, {[k]=dump(v)})
            end
            return d
        elseif types.is_type(m, List) then
            local d = {'__List__'}
            for i, v in ipairs(m) do
                table.insert(d, dump(v))
            end
            return d
        else
            if kwargs.keep_tensor and torch.isTensor(m) then
                return m
            elseif kwargs.keep_scalar and (type(m) == 'string' or type(m) == 'number') then
                return m
            else
                return 0
            end
        end
    end
    return dump(map)
end
function nested.loadfromtable(tb)
    local function ld(t)
        if type(t) ~= 'table' then
            return t
        else
            if t[1] == '__Om__' then
                local d = {}
                for i=2,#t do
                    table.insert(d, {[tablex.keys(t[i])[1]]=ld(tablex.values(t[i])[1])})
                end
                return Om(d)
            elseif t[1] == '__List__' then
                local d = {}
                for i=2,#t do
                    table.insert(d, ld(t[i]))
                end
                return List(d)
            else
                error('unexpected head: ' .. (t[1] or 'nil'))
            end
        end
    end
    return ld(tb)
end
return nested