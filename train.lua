local train = {}
--[[important global variables]]
-- experiment settings and other options
__opt = nil
-- prototypes of models, to be cloned
__protos = nil
-- clones of prototypes
__clones = nil
-- experiment state and statistics, such as losses
__xstate = nil
-- optimizer history, such as learning rate
__optim_hist = nil
-- rnn outputs during each run, such as probability distribution of symbols,
-- and key/memory vectors.
__rnn_states = nil
-- gradients during backprop
__d_states = nil
-- optimizer state, such as learning rate (and only learning rate right now)
__optim_state = nil
-- pooled parameter and its gradient, used for blackbox optimization
__params = nil
__grad_params = nil
-- state of train/valid dataset; contains pointer and statistics
__state_train = nil
__state_valid = nil
-- module responsible for forward computation and backprop
__fb = nil
-- module that contains constructor and initializer for the desired model
__model = nil

function train.imports()
    require 'nn'
    require 'nngraph'
    require 'totem'
    require 'optim'
    require 'paths'
    require 'pl'
    cjson = require 'cjson'
    require('pl.stringx').import()
    Om = OrderedMap

    options = dofile('options.lua')
    -- nngraph.setDebug(true)

    ioutils = dofile('utils/io.lua')
    utils = dofile('utils/utils.lua')

    for f in paths.files('layers', 'lua') do
        dofile('layers/'..f)
    end

    dofile('kwrap/init.lua')

    checkpoint = dofile('utils/checkpoint.lua')
    loader = dofile('data/load.lua')
    pp = dofile('utils/pp.lua')
    pp_om = pp.pp_om

end

function train.gpu()
    -- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
    if __opt.gpuid >= 0 and __opt.opencl == 0 then
        local ok, _ = pcall(require, 'cunn')
        local ok2, _ = pcall(require, 'cutorch')
        if not ok then print('package cunn not found!') end
        if not ok2 then print('package cutorch not found!') end
        if ok and ok2 then
            print('using CUDA on GPU ' .. __opt.gpuid .. '...')
            cutorch.setDevice(__opt.gpuid + 1) -- +1 to make it 0 indexed
            cutorch.manualSeed(__opt.seed)
        else
            print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
            print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
            print('Falling back on CPU mode')
            __opt.gpuid = -1 -- overwrite user setting
            cunn = nil
            cutorch = nil -- cutorch is later used to test whether using gpu, so this needs to be nil
            print('cunn is nil', cunn == nil)
            print('cutorch is nil', cutorch == nil)
        end
    end
end

function train.process_data()
    local _train, _valid
    _train, _valid, loader.idx2char, loader.char2idx = loader.load(__opt)
    __state_train = {data=utils.gpu(_train:t(), __opt)}
    __state_valid =  {data=utils.gpu(_valid:t(), __opt)}
    local states = {__state_train, __state_valid}
    for _, state in pairs(states) do
        train.reset_state(state, __protos, __opt)
    end
end

function train.load_or_create()
    train.process_data()
    if string.len(__opt.use_checkpoint) > 0 then
        print('loading checkpoint', __opt.use_checkpoint)
        __opt, __protos, __xstate, __optim_hist, __checkpoint_extra =
            checkpoint.load_checkpoint(__opt.use_checkpoint, __opt.checkpoint_load_mode, __opt)
        __using_checkpoint = true
    else
        print('initializing fresh model parameters')
        __protos = train.init_protos(__opt)
        __using_checkpoint = false
        __checkpoint_extra = nil
    end
end

function train.init_protos(opt)
    __model = dofile('models/' .. opt.model .. '.lua')
    local protos = {}
    opt.vocab_size = tablex.size(loader.idx2char)
    local weights = utils.gpu(torch.ones(opt.vocab_size), opt)
    protos.criterion = nn.CrossEntropyCriterion(weights)
    protos.enc_dict = nn.LookupTable(opt.vocab_size, opt.embedding_size)
    protos.rnn = __model.model(opt, {enc_dict=protos.enc_dict, criterion=protos.criterion})
    protos.init_states = __model.learnable_init_states(opt)
    for _, proto in pairs(protos) do
        utils.gpu(proto, opt, true)
    end
    return protos
end

function train.reset_acc(state)
    state.coarse = 0
    state.coarse_total = 0
    state.coarse_acc = -1
    state.fine = 0
    state.fine_total = 0
    state.fine_acc = -1
end
function train.reset_state(state, protos, opt)
    state.pos = 1
    train.reset_acc(state)
end

function train.pool_params()
    __params, __grad_params = utils.combineAllParameters(__protos.rnn, __protos.init_states)
    print('Total number of params:', __params:nElement())
end

function train.reinit_params(magni, kwargs)
    kwargs = kwargs or {}
    forget_bias = kwargs.forget_bias or __opt.forget_bias
    in_bias = kwargs.in_bias or __opt.in_bias
    out_bias = kwargs.out_bias or __opt.out_bias
    if forget_bias == 'init_magnitude' then
        forget_bias = nil
    end
    if in_bias == 'init_magnitude' then
        in_bias = nil
    end
    if out_bias == 'init_magnitude' then
        out_bias = nil
    end

    if __using_checkpoint then
        print('using checkpoint, no reinit')
        return
    end
    magni = magni or __opt.init_magnitude
    if magni ~= false then
        print('Reinitializing weights to magnitude', magni)
        print('param before reinit has mean', __params:mean(), 'and variance', __params:var())
        __params:uniform(-magni, magni)
        for _, node in ipairs(__protos.rnn.mod:findModules('nn.LSTMGatesX')) do
            -- division by root(3) to offset the mult. by root(3) in nn.Linear:reset
            node:reset(magni / math.sqrt(3), forget_bias, in_bias, out_bias)
        end
        print('param after reinit has mean', __params:mean(), 'and variance', __params:var())
    else
        print('Not reinitializing weights')
    end
end
function train.clone()
    --[[Cloning rnn across time]]
    local timer = torch.Timer()
    __clones = {}
    local rnn_num
    if __opt.task == 'addition' then
        rnn_num = 3 * __opt.max_valid_seq_len + 5
    elseif __opt.task == 'repeatCopy' then
        rnn_num = (__opt.valid_nrepeat_high+1) * __opt.max_valid_seq_len
                    + __opt.valid_nrepeat_high + 5
    elseif __opt.task == 'prioritySort' then
        rnn_num = __opt.max_valid_seq_len * (__opt.max_valid_seq_len - 1) / 2 -- priorities
                + __opt.max_valid_seq_len * 2 + 5 -- everything else
    else
        rnn_num = 2 * __opt.max_valid_seq_len + 3
    end
    __clones.rnn = utils.cloneManyTimesFast(__protos.rnn, rnn_num)
    local time = timer:time().real

    print('Cloning took', time, 'seconds')
end

function train._init_xstate()
    local xstate = {}
    xstate.best_loss = nil
    xstate.train_losses = {}
    xstate.train_fine_acc = {}
    xstate.train_coarse_acc = {}
    xstate.valid_losses = {}
    xstate.valid_fine_acc = {}
    xstate.valid_coarse_acc = {}
    xstate.iter = 1
    return xstate
end

function train._init_optim_state()
    local optim_state = {learningRate=__opt.learning_rate}
    return optim_state
end

function train.init_metadata()
    if __using_checkpoint ~= true then
        print('no checkpoint; init fresh xstate and optim_hist')
        __xstate = train._init_xstate()
        __optim_hist = {}
        __optim_state = train._init_optim_state()
        __optim_hist[__xstate.iter] = tablex.deepcopy(__optim_state)
    else
        print('using checkpoint')
        if not __xstate then
            print('init fresh xstate')
            __xstate = train._init_xstate()
        end
        if not __optim_hist then
            print('init fresh optim_hist')
            __optim_hist = {}
            __optim_state = train._init_optim_state()
            __optim_hist[__xstate.iter] = tablex.deepcopy(__optim_state)
        elseif #__optim_hist > 0 then
            print('nonempty optim_hist; setting optim_state')
            __optim_state = tablex.deepcopy(__optim_hist[#__optim_hist])
        else
            print('empty optim_hist; setting optim_state')
            __optim_state = train._init_optim_state()
        end
    end
    -- keeps track of total time spent; is instantiated at first call of looptrain()
    __total_timer = nil
end

function train.feval(params_)
    -- return average loss per symbol and average grad per symbol
    if __params ~= params_ then
        __params:copy(params_)
    end
    __grad_params:zero()

    local forward_states = __fb.forward(__state_train, __opt, __protos, __clones)
    local rnn_states = forward_states.rnn_states
    -- average loss per char tested
    local loss = forward_states.loss / forward_states.nloss

    __fb.backward(__state_train, rnn_states, __opt, __protos, __clones)

    __grad_params:div(forward_states.nloss)
    local grad_norm = __grad_params:norm()
    if grad_norm > __opt.max_grad_norm then
        local scaling_factor = __opt.max_grad_norm / grad_norm
        __grad_params:mul(scaling_factor)
    end
    return loss, __grad_params
end
function train.run_(state)
    --[[run one single pass through the dataset `state`]]
    train.reset_state(state, __protos, __opt)
    utils.changeRnnMode(__clones.rnn, false)
    local loss = 0
    local nloss = 0
    while true do
        local forward_states, cycle = __fb.forward(state, __opt, __protos, __clones)
        loss = loss + forward_states.loss
        nloss = nloss + forward_states.nloss
        if cycle then
            break
        end
    end
    utils.changeRnnMode(__clones.rnn, true)
    return loss / nloss
end
function train.run_valid()
    return train.run_(__state_valid)
end

-- record memory access for visualization
function train.run_vis()
    if not (__opt.num_memory_modules > 0 and __opt.model:endswith('lantm')) then
        return
    end
    __xstate.vis = __xstate.vis or {keys={}, keys_read={}, mem_strength={}}
    local state = __state_valid
    train.reset_state(state, __protos, __opt)
    utils.changeRnnMode(__clones.rnn, false)
    __fb.forward(state, __opt, __protos, __clones)
    utils.changeRnnMode(__clones.rnn, true)

    -- __rnn_states comes from __fb exposing rnn_states during forward
    local keys = __rnn_states[#__rnn_states].memory[1].keys_
    -- print(keys)
    local w = 1
    local function split_key(which)
        local function s(k)
            return k[{w, which}]
        end
        return s
    end
    local keys_x = tablex.map(split_key(1), keys)
    local keys_y = tablex.map(split_key(2), keys)
    local which_head = 1
    local _, keys_read = utils.stateslice(__rnn_states,
        {'memory', 1, 'key_read', which_head, 1})
    keys_read = torch.Tensor(keys_read)
    local rng = {}
    local keys_read_x = torch.totable(keys_read[{rng, 1}]:float())
    local keys_read_y = torch.totable(keys_read[{rng, 2}]:float())

    local keys_read_z, keys_z
    if __opt.model == 'slantm' then
        keys_z = tablex.map(split_key(3), keys)
        keys_read_z = torch.totable(keys_read[{rng, 3}]:float())
    end

    local mem_strength = tablex.map(function(x) return x[1][1] end,
            __rnn_states[#__rnn_states].memory[1].mem_strength)

    __xstate.vis.keys[__xstate.iter] = {x=keys_x, y=keys_y, z=keys_z}
    __xstate.vis.keys_read[__xstate.iter] = {
                            x=keys_read_x, y=keys_read_y, z=keys_read_z}
    __xstate.vis.mem_strength[__xstate.iter] = mem_strength
end

function train.valid_save(valid, kwargs)
    kwargs = kwargs or {}
    local save_chkpt = kwargs.save_chkpt
    local name = kwargs.name
    if valid then
        print('validating...')
        local timer = torch.Timer()
        __xstate.valid_losses[__xstate.iter] = train.run_valid()
        __xstate.valid_fine_acc[__xstate.iter] = __state_valid.fine_acc
        __xstate.valid_coarse_acc[__xstate.iter] = __state_valid.coarse_acc
        local time = timer:time().real
        print(string.format('validation loss %.3f', __xstate.valid_losses[__xstate.iter]))
        print(string.format('validation fine acc %.1f', __state_valid.fine_acc * 100))
        print(string.format('validation coarse acc %.1f', __state_valid.coarse_acc * 100))
        print('validation took', time / 60, 'min')
    end

    local f1 = function (x) return string.format('%.1f', x * 100) end
    local tloss = torch.Tensor(__xstate.train_losses)
    local chpt_name = ('iter'..__xstate.iter..
        'nsamples'..__xstate.iter * __opt.iter_size * __opt.batch_size)
    if __xstate.train_losses[__xstate.iter] then
        chpt_name = chpt_name .. '_train-' ..
            __xstate.train_losses[__xstate.iter] .. '-' ..
            __xstate.train_fine_acc[__xstate.iter] .. '-' ..
            __xstate.train_coarse_acc[__xstate.iter]
    end
    if name then
        chpt_name = name .. '_' .. chpt_name
    end
    if valid then
        chpt_name = chpt_name .. '_valid' .. f1(__xstate.valid_coarse_acc[__xstate.iter])
    end
    if save_chkpt then
        checkpoint.save_checkpoint(chpt_name, __opt, __protos, __xstate, __optim_hist)
    else
        print('not saving checkpoint')
    end
end

function train.looptrain()
    __total_timer = __total_timer or torch.Timer()
    utils.changeRnnMode(__clones.rnn, true)
    local time
    local timer = torch.Timer()
    local L = 0
    train.reset_acc(__state_train)
    for i = 1, __opt.iter_size do
        local _, loss = optim[__opt.optimizer](train.feval, __params, __optim_state)
        L = L + loss[1]
    end
    -- loss per char
    __xstate.train_losses[__xstate.iter] = L / __opt.iter_size
    __xstate.train_fine_acc[__xstate.iter] = __state_train.fine_acc
    __xstate.train_coarse_acc[__xstate.iter] = __state_train.coarse_acc
    __optim_hist[__xstate.iter] = {
        learningRate=__optim_state.learningRate}
    time = timer:time().real

    local report_interval = __opt.progress_report_interval
    if __xstate.iter %  report_interval == 0 then
        local report = os.date('%c') .. ' '
        report = report .. string.format(
                'total time %.1fm, iter %d, nsamples %.3e, '..
                'tloss %.3f, tfine %.1f, tcoarse %.1f, ' ..
                'time/iter %.2fs, time/char %4.4es, ' ..
                'train.pos %d',
                __total_timer:time().real / 60, __xstate.iter,
                __xstate.iter * __opt.iter_size * __opt.batch_size,
                __xstate.train_losses[__xstate.iter],
                __xstate.train_fine_acc[__xstate.iter] * 100,
                __xstate.train_coarse_acc[__xstate.iter] * 100,
                -- fine_total is #(char tested) in this batch, which is linearly related
                -- to #(char read) in this batch
                time, time / __state_train.fine_total,
                __state_train.pos)
        print(report)
        if __xstate.iter > __opt.decay_delay then
            __optim_state.learningRate =
                __optim_state.learningRate * __opt.decay_rate
        end
        if __xstate.iter % __opt.valid_interval == 0 then
            if __xstate.iter > __opt.valid_after and
                    __xstate.valid_coarse_acc[__xstate.iter] == nil then
                local save_chkpt = __xstate.iter % __opt.checkpoint_freq == 0 or
                    __xstate.iter == __opt.max_iter
                train.valid_save(true,
                    {save_chkpt=save_chkpt})
            end
        end
        if __xstate.iter % __opt.reset_train_interval == 0 then
            train.reset_state(__state_train, __protos, __opt)
        end
    end
    if __opt.run_vis then
        train.run_vis()
    end
    if __xstate.iter % 33 == 0 then
        if cutorch then
            cutorch.synchronize()
        end
        collectgarbage()
    end

    __xstate.iter = __xstate.iter + 1
end
return train
