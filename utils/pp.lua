local M = {}
function M.sizestr(x)
    local strt = {}
    if _G.torch.typename(x):find('torch.*Storage') then
        return _G.torch.typename(x):match('torch%.(.+)') .. ' - size: ' .. x:size()
    end
    if x:nDimension() == 0 then
        table.insert(strt, _G.torch.typename(x):match('torch%.(.+)') .. ' - empty')
    else
        table.insert(strt, _G.torch.typename(x):match('torch%.(.+)') .. ' - size: ')
        for i=1,x:nDimension() do
            table.insert(strt, x:size(i))
            if i ~= x:nDimension() then
                table.insert(strt, 'x')
            end
        end
    end
    return table.concat(strt)
end

function M.pp_om(map, mode)
    local function pp(m, leading)
        if tablex.find({'Map', 'OrderedMap'}, types.type(m)) then
            for k, v in m:iter() do
                print(leading .. k)
                pp(v, leading .. '\t')
            end
        elseif types.is_type(m, List) then
            for k, v in ipairs(m) do
                print(leading .. k)
                pp(v, leading .. '\t')
            end
        elseif mode ~= 'keys' then
            if torch.isTensor(m) then
                local s = leading .. M.sizestr(m)
                if m.norm then
                    s = s .. '\tnorm ' .. m:norm()
                end
                print(s)
            elseif type(m) == 'number' or type(m) == 'boolean' then
                print(leading .. tostring(m))
            else
                print(leading .. torch.type(m))
            end
        end
    end
    pp(map, '')
end

function M.tb(t)
    -- thanks to https://coronalabs.com/blog/2014/09/02/tutorial-printing-table-contents/
    local print_r_cache={}
    local function sub_print_r(t,indent)
        if (print_r_cache[tostring(t)]) then
            print(indent.."*"..tostring(t))
        else
            print_r_cache[tostring(t)]=true
            if (type(t)=="table") then
                for pos,val in pairs(t) do
                    if (type(val)=="table") then
                        print(indent.."["..pos.."] => {")
                        sub_print_r(val,indent..string.rep(" ",string.len(pos)+8))
                        print(indent..string.rep(" ",string.len(pos)+6).."}")
                    elseif (type(val)=="string") then
                        print(indent.."["..pos..'] => "'..val..'"')
                    else
                        print(indent.."["..pos.."] => "..tostring(val))
                    end
                end
            else
                print(indent..tostring(t))
            end
        end
    end
    if (type(t)=="table") then
        print(tostring(t).." {")
        sub_print_r(t,"  ")
        print("}")
    else
        sub_print_r(t,"  ")
    end
    print()
end


return M