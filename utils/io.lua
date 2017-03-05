require 'paths'
local io = {}
local common = paths.dofile('common.lua')
function io.read_file(filepath)
    local file = torch.DiskFile(filepath, 'r')
    return file:readString('*a')
end

function io.read_json(filepath)
    local cjson = require 'cjson'
    local c = io.read_file(filepath)
    return cjson.decode(c)
end

function io.write_str(filepath, s)
    local file = torch.DiskFile(filepath, 'w')
    file:writeString(s)
    file:close()
end

function io.write2json(filepath, d, keys2str)
    d = keys2str and common.keys2str(d) or d
    io.write_str(filepath, cjson.encode(d))
end

function io.saveh5(m, file, path, silent)
    require 'hdf5'
    silent = silent or true
    if types.is_type(m, Om) or types.is_type(m, Map) then
        for k, v in m:iter() do
            io.saveh5(v, file, path .. '/' .. k)
        end
    elseif types.is_type(m, List) then
        for i, v in ipairs(m) do
            io.saveh5(v, file, path .. '/' .. i)
        end
    elseif torch.isTensor(m) then
        local status, err = pcall(function() file:write(path, m:double()) end)
        if not silent then
            if not status then
                print('error writing to path', path)
                print(err)
            else
                print('wrote to path', path)
            end
        end
    else
        error('unexpected type')
    end
end
return io