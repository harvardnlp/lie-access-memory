require 'paths'
require 'hdf5'
require 'cjson'

local function read_file(filepath)
    local file = torch.DiskFile(filepath, 'r')
    return file:readString('*a')
end

local loader_path = paths.dirname(paths.thisfile())
local loader = {}
function loader.get_translate(file)
  local f = io.open(file,'r')
  local idx2char = {}
  local char2idx = {}
  for line in f:lines() do
    local c = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(c, w)
    end
    idx2char[tonumber(c[2])] = c[1]
    char2idx[c[1]] = tonumber(c[2])
  end
  return idx2char, char2idx
end
function loader.load(opt)
  local metapath = paths.concat(opt.dataset, 'metadata.json')
  local meta = cjson.decode(read_file(metapath))

  local tensorfile = paths.concat(opt.dataset, 'tensor.hdf5')
  local tensors = hdf5.open(tensorfile, 'r')
  local train = tensors:read('train'):all()
  local valid = tensors:read('valid'):all()
  tensors:close()

  for k, v in pairs(meta) do
    opt[k] = v
  end
  local idx2char, char2idx = loader.get_translate(
    paths.concat(opt.dataset, 'str2int.txt'))

  return train, valid, idx2char, char2idx
end

return loader
