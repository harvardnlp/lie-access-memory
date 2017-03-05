require 'paths'
require 'pl'
paths.dofile('../kwrap/kSerial.lua')
local utils = paths.dofile('utils.lua')
local Om = OrderedMap

local checkpoint = {}
local function read_opt(filepath)
    local file = torch.DiskFile(filepath, 'r')
    return file:readString('*a')
end
local function write_opt(filepath, opt)
    local file = torch.DiskFile(filepath, 'w')
    file:writeString(opt)
    file:close()
end
function checkpoint.save_checkpoint(str, opt, protos, xstate, optim_hist, kwargs)
    --[[Create folder name based on options. If folder does not exist,
    create folder and store a copy of `opt` in text as `folder/.opt`, and then
    serialize and store as `folder/[checkpointname]'.
    If folder exists, check `folder/.opt` matches `opt`, and if so
    serialize and store. Otherwise, raise error.]]
    kwargs = kwargs or {}
    str = str or ''
    local savedir = paths.concat(opt.checkpoint_dir, opt.expname, opt.exp_id)
    paths.mkdir(savedir)
    local saved_json = nil
    local opt_path = paths.concat(savedir, '.opt')
    if not pcall(function() saved_json = read_opt(opt_path) end) then
        print('.opt file not found at'..savedir)
        print('creating new .opt')
        local opt_json = cjson.encode(opt)
        write_opt(opt_path, opt_json)
    else
        local old_opt = cjson.decode(saved_json)
        for k, v in pairs(old_opt) do
            local opt_eq = tablex.deepcompare(old_opt[k], opt[k])
            assert(opt_eq, 'options does not match .opt')
        end
    end
    local filename = string.format('%s.t7', str)
    local savefile = paths.concat(savedir, filename)
    local checkpoint = {}
    checkpoint.protos = {}
    protos_kmaps = {}
    for k, p in pairs(protos) do
        local _, serial = pcall(function()
                    return nn.kSerial(p)
                end)
        if type(serial) == 'string' then
            if serial:lfind('no implementation') then
                checkpoint.protos[k] = p
            else
                error(serial)
            end
        else
            print('saving protos', k)
            checkpoint.protos[k] = serial
        end
    end
    checkpoint.opt = opt
    checkpoint.xstate = xstate
    checkpoint.optim_hist = optim_hist
    checkpoint.rng_state = torch.getRNGState()
    if cutorch then
        checkpoint.cuda_rng_state = cutorch.getRNGState()
    end
    torch.save(savefile, checkpoint)
    print('saved checkpoint at ' .. savefile)
    return savefile
end

function checkpoint.load_checkpoint(
    chkpt, mode, opt, protos, xstate, optim_hist, extra)
    --[[chkpt cannot be nil or '']]
    if not chkpt or chkpt and string.len(chkpt) == 0 then
        error("chkpt cannot be nil or ''")
    end
    print('Loading checkpoint from ' .. chkpt)
    local checkpoint = torch.load(chkpt)
    for k, p in pairs(checkpoint.protos) do
        if torch.isTypeOf(p, nn.kSerial) then
            checkpoint.protos[k] = p:load()
        end
    end
    protos = protos or {}
    tablex.update(protos, checkpoint.protos)
    if mode == nil or mode == '' or mode == 'model_params' then
        print('Overwriting model params to checkpoint values.')
        opt = opt or {}
        for _, p in ipairs(options.O.model:keys()) do
            opt[p] = checkpoint.opt[p]
        end
    elseif mode == 'complete' then
        print('Reload all options and state from checkpoint')
        opt = opt or {}
        tablex.update(opt, checkpoint.opt)
        extra = extra or {}
        if checkpoint.xstate then
            xstate = xstate or {}
            tablex.update(xstate, checkpoint.xstate)
        end
        if checkpoint.optim_hist then
            optim_hist = optim_hist or {}
            tablex.update(optim_hist, checkpoint.optim_hist)
        end
        if checkpoint.rng_state then
            torch.setRNGState(checkpoint.rng_state)
        end
        if cutorch and checkpoint.cuda_rng_state then
            cutorch.setRNGState(checkpoint.cuda_rng_state)
        end
    end
    if opt.gpuid and opt.opencl then
        for _, proto in pairs(protos) do
            pcall(function () utils.gpu(proto, opt) end)
        end
    end
    return opt, protos, xstate, optim_hist, extra
end

return checkpoint