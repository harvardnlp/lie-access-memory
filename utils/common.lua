
local common = {}
function common.get_current_git_hash()
    return sys.execute('git rev-parse HEAD'):strip()
end
function common.get_current_git_branch()
    return sys.execute('git rev-parse --abbrev-ref HEAD'):strip()
end

function common.gpu(x, opt, nothrow)
    nothrow = nothrow or false
    if opt.gpuid >= 0 and opt.opencl == 0 then
        if not nothrow then
            return x:cuda()
        else
            if type(x) == 'function' then
                return x
            else
                return x.cuda and x:cuda() or x
            end
        end
    elseif opt.gpuid >= 0 and opt.opencl == 1 then
        if not nothrow then
            return x:cl()
        else
            return x.cl and x:cl() or x
        end
    else
        return x
    end
end

function common.cpu(x)
    return x:double()
end

function common.gpuzeros(opt)
    local function zeros(...)
        return common.gpu(torch.zeros(...), opt)
    end
    return zeros
end

function common.convert2Reals(tensor)
    if __opt.gpuid >= 0 and __opt.opencl == 0 then
        return tensor:cuda()
    else
        return tensor:double()
    end
end


function common.changeRnnMode(rnn, is_training)
    for i = 0, #rnn do
        if is_training then
            rnn[i]:training()
        else
            rnn[i]:evaluate()
        end
    end
end

function common.rfind(s, pattern, start, end_)
    s = s:sub(start, end_)
    return s:len() - s:reverse():find(pattern) + 1
end

function common.isnan(x)
    if type(x) == 'number' then
        return x ~= x
    elseif torch.isTensor(x) then
        return not x:eq(x):all()
    end
end

function common.isfinite(x)
    return x < math.huge and x > -math.huge
end

function common.frac(tensor)
    return tensor - torch.floor(tensor)
end

function common.rev_iter(iter)
    return List(iter):reverse():iter()
end

function common.escapestr(s, mode)
    mode = mode or 'postgresql'
    if mode == 'postgresql' then
        return s:replace('\'', '\'\'')
    end
end
function common.everyother(a)
    local stride = a:stride()
    local size = a:size()
    stride[#stride] = 2
    size[#size] = size[#size] / 2
    local b = torch.Tensor(a:storage(), 1, size, stride)
    local c = torch.Tensor(a:storage(), 2, size, stride)
    return b, c
end
function common.listfill(k, stuff, call)
    local l = List()
    for i=1,k do
        l:append(call and stuff() or stuff)
    end
    return l
end
function common.listinput(k)
    require 'nn'
    local l = List()
    for i=1,k do
        l:append(nn.Identity()())
    end
    return l
end

function common.keys2str(t)
    -- convert recursively all keys of a (nested) table to string
    -- (primarily for json serialization)
    if type(t) == 'table' then
        local s = {}
        for k, v in pairs(t) do
            s[tostring(k)] = common.keys2str(v)
        end
        return s
    else
        return t
    end
end

function common.rand_int_table(dim_low, dim_high, entry_low, entry_high)
    -- returns a table with number of entries in [dim_low, dim_high)
    -- each entry is in the range [entry_low, entry_high)
    dim_low = dim_low or 2
    dim_high = dim_high or 7
    entry_low = entry_low or 1
    entry_high = entry_high or 10
    local ndim = torch.random() % (dim_high - dim_low) + dim_low
    local size = torch.totable(torch.floor(torch.rand(ndim) * (entry_high - entry_low)) + entry_low)
    return size
end

function common.combineAllParameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

-- Authors: Tomas Kocisky
-- (Modification of above.)
function common.cloneManyTimesFast(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end
   local paramsNoGrad
   if net.parametersNoGrad then
      paramsNoGrad = net:parametersNoGrad()
   end
   local mem = torch.MemoryFile("w"):binary()
   mem:writeObject(net)

   -- serialize an empty clone for faster cloning
   do
      local reader = torch.MemoryFile(mem:storage(), "r"):binary()
      local clone = reader:readObject()
      reader:close()
      if net.parameters then
          local cloneParams, cloneGradParams = clone:parameters()
          local cloneParamsNoGrad
          for i = 1, #params do
             cloneParams[i]:set(torch.Tensor():typeAs(params[i]))
             cloneGradParams[i]:set(torch.Tensor():typeAs(gradParams[i]))
          end
          if paramsNoGrad then
             cloneParamsNoGrad = clone:parametersNoGrad()
             for i =1,#paramsNoGrad do
                cloneParamsNoGrad[i]:set(torch.Tensor():typeAs(paramsNoGrad[i]))
             end
          end
      end

      mem:close()
      mem = torch.MemoryFile("w"):binary()
      mem:writeObject(clone)
   end
   collectgarbage()
   for t = 0, T-1 do
      -- We need to use a new reader for each clone.
      -- We don't want to use the pointers to already read objects.
      local reader = torch.MemoryFile(mem:storage(), "r"):binary()
      local clone = reader:readObject()
      reader:close()
      if net.parameters then
          local cloneParams, cloneGradParams = clone:parameters()
          local cloneParamsNoGrad
          for i = 1, #params do
             cloneParams[i]:set(params[i])
             cloneGradParams[i]:set(gradParams[i])
          end
          if paramsNoGrad then
             cloneParamsNoGrad = clone:parametersNoGrad()
             for i =1,#paramsNoGrad do
                cloneParamsNoGrad[i]:set(paramsNoGrad[i])
             end
          end
      end
      clones[t] = clone
      -- My modification
      if torch.isTypeOf(net, 'nn.kModule') or torch.isTypeOf(net, 'nn.kWrap') then
        clone.input_map = net.input_map
        clone.output_map = net.output_map
      end
      collectgarbage()
   end
   mem:close()
   return clones
end

return common
