require 'torch'
require 'nn'
require 'nngraph'
require 'pl'
require 'paths'
paths.dofile('../layers/VAddTable.lua')
paths.dofile('../layers/CMulList.lua')
paths.dofile('../layers/InvNormalize.lua')
paths.dofile('../layers/PowSum.lua')
paths.dofile('../layers/Append.lua')
paths.dofile('../kwrap/kModule.lua')
paths.dofile('../layers/Narrow.lua')
local utils = paths.dofile('../utils/utils.lua')
local st = paths.dofile('../kwrap/split_transfer.lua')
local Om = OrderedMap
local pp = paths.dofile('../utils/pp.lua')
local pp_om = pp.pp_om
local listfill = utils.listfill

local lantm = {}

local function _mix(gate, key_t, key_tm1)
    -- gate = 1 => key_t
    -- gate = 0 => key_tm1
    local _mix_tensor = nn.JoinTable(1, 1){gate, nn.OneMinus()(gate)}
    local _both_keys = nn.ParallelTable()
    _both_keys:add(nn.Identity())
    _both_keys:add(nn.Identity())
    -- new key should have size (batch_size, key_dim)
    nngraph.annotateNodes()
    return nn.MixtureTable(3){_mix_tensor, _both_keys{key_t, key_tm1}}
end

local function _mix_softmax(gates, key_t, key_tm1, key_storage)
    -- gates[batch, 1] should be the coefficient for key_t
    -- gates[batch, 2] should be the coefficient for key_tm1
    local allkeys = nn.JoinTable(2){
        nn.View(1, -1):setNumInputDims(1)(key_t),
        nn.View(1, -1):setNumInputDims(1)(key_tm1),
        key_storage}
    return nn.MixtureTable(2){gates, allkeys}
end
function lantm.mix_keys(key_t, key_tm1, gate, action, opt, extra)
    -- key_tm1 and key_t have size (batch_size, key_dim)
    -- gate has size (batch_size, 1)
    -- action has size (batch_size, key_dim)
    --[[ returns mixed key plus mixed action
    ]]

    -- _new_key should have size (batch_size, key_dim)
    local _new_key = _mix(gate, key_t, key_tm1)
    if extra and extra.act_tm1 then
        if not opt.act_interp then
            extra.gate_act = nn.AddConstant(1)(nn.MulConstant(0)(extra.gate_act))
        end
        action = _mix(extra.gate_act,
                    action,
                    extra.act_tm1)
    end
    local step = nn.CAddTable(){_new_key, action}

    nngraph.annotateNodes()
    return step, action
end

function lantm.mem_step(mem_tm1, keys_tm1, key_write_t, key_read_t, write_vec_t, opt, extra)
    -- mem_tm1 has size (batch_size, mem_length, mem_width)
    -- keys_tm1 has size (batch_size, mem_length, key_dim)
    -- key_write_t and key_read_t have size (batch_size, key_dim); they should be normalized
    -- write_vec_t has size (batch_size, mem_width)

    -- read_weights_t has size (batch_size, mem_length)
    local read_weights_t = lantm.get_read_weights(keys_tm1, key_read_t, opt, extra)
    local read_t
    -- if mem_step is called through controller, then key_read_t should be a list
    if types.is_type(read_weights_t, List) then
        read_t = List()
        for i=1,opt.num_read do
            read_t:append(nn.MixtureTable(){read_weights_t[i], nn.Identity()(mem_tm1)})
        end
    else
        -- read_t has size (batch_size, mem_width)
        read_t = nn.MixtureTable(){read_weights_t, nn.Identity()(mem_tm1)}
    end


    local mem_t
    if write_vec_t then
        local _read_t
        if type(read_t) == 'table' and #read_t == 1 then
            _read_t = read_t[1]
        end
        mem_t = nn.Append(){mem_tm1, write_vec_t, _read_t}
    else
        mem_t = mem_tm1
    end
    local _read_weights_t
    if type(read_weights_t) == 'table' and #read_weights_t == 1 then
        _read_weights_t = read_weights_t[1]
    end
    local keys_t
    if key_write_t then
        keys_t = nn.Append(){keys_tm1, key_write_t, _read_weights_t}
    else
        keys_t = keys_tm1
    end

    local mem_strength_t
    if extra and extra.write_strength_t then
        -- write_strength_t should have dimension (batch_size, 1)
        mem_strength_t = nn.Append(){extra.mem_strength_tm1,
                                    extra.write_strength_t,
                                    _read_weights_t}
    end

    nngraph.annotateNodes()
    local ret = OrderedMap{
        {mem_t=mem_t}, {keys_t=keys_t}, {read_t=read_t},
        {read_weights_t=read_weights_t},
        {mem_strength_t=mem_strength_t}}
    return ret
end
function lantm.get_read_weights(keys_tm1, key_read_t, opt, extra)
    if types.is_type(key_read_t, List) then
        local rw = List()
        for i=1,opt.num_read do
            local extra_ = tablex.copy(extra)
            extra_.temperature = extra_.temperature and extra_.temperature[i]
            rw:append(lantm._get_read_weights(keys_tm1, key_read_t[i], opt, extra_))
        end
        return rw
    else
        return lantm._get_read_weights(keys_tm1, key_read_t, opt, extra)
    end
end
function lantm._get_read_weights(keys_tm1, key_read_t, opt, extra)
    local ds
    local diff = nn.VAddTable(true){key_read_t, keys_tm1}
    ds = nn.PowSum(opt.sumpow, 3)(diff)
    if extra and extra.temperature and opt.weight_method == 'SoftMax' then
        -- temperature has size (batch_size, 1)
        -- ds <- ds/softplus(temperature)
        ds = nn.CDivTable(){ds,
            nn.ExpandAs(){
                nn.Reshape(1, true)(
                    nn.Sum(2)(
                        nn.SoftPlus()(
                            extra.temperature
                            )
                        )
                    ),
                ds}
            }
    end

    local read_weights
    if opt.weight_method == 'SoftMax' then
        read_weights = nn.SoftMax()(nn.Minus()(ds))
    elseif opt.weight_method == 'InvNormalize' then
        read_weights = nn.InvNormalize(opt.invnorm_pow)(ds)
    else
        error('unknown weight_method')
    end
    -- mem_strength_tm1 has shape (batch_size, mem_length)
    read_weights = nn.CMulList(){extra.mem_strength_tm1, read_weights}
    read_weights = nn.Normalize(1)(
        nn.AddConstant(opt.epsilon_read_weight, true)(
            read_weights
        )
    )
    nngraph.annotateNodes()
    return read_weights
end
function lantm._model(opt, kwargs)
    -- convenience variables
    local dropout = opt.dropout or 0
    local n = opt.num_lstm_layers
    local m = opt.num_memory_modules
    local msz = opt.mem_width
    local rsz = opt.rnn_size
    local esz = opt.embedding_size
    local vsz = opt.vocab_size
    local ksz = opt.key_dim

    local listfill = function(stuff) return utils.listfill(opt.num_read, stuff) end
    local listinput = utils.listinput

    local inputs = Om()
    inputs.x = nn.Identity()()
    inputs.controller = Om()
    for L = 1, n do
        inputs.controller[L] = Om()
        inputs.controller[L].lstm_c = nn.Identity()()
        inputs.controller[L].lstm_h = nn.Identity()()
    end
    inputs.memory = Om()
    for s = 1, m do
        inputs.memory[s] = Om()
        inputs.memory[s].mem = nn.Identity()()
        inputs.memory[s].keys_ = nn.Identity()()
        inputs.memory[s].key_read = listinput(opt.num_read)
        inputs.memory[s].key_write = nn.Identity()()
        inputs.memory[s].read_val = listinput(opt.num_read)
        inputs.memory[s].mem_strength = nn.Identity()()
        inputs.memory[s].act_read = listinput(opt.num_read)
        inputs.memory[s].act_write = nn.Identity()()
    end

    local outputs = Om()
    outputs.controller = Om()
    outputs.memory = Om()

    local x, input_size_L

    for L = 1, n do
        -- cell and hidden state from previous timesteps
        local prev_c = inputs.controller[L].lstm_c
        local prev_h = inputs.controller[L].lstm_h

        -- the input to this layer
        if L == 1 then
            if opt.num_memory_modules == 0 then
                x = inputs.x
                input_size_L = esz
            else
                local controller_inputs = List{ inputs.x }
                input_size_L = esz
                -- collect all reads from previous step
                for s = 1, m do
                    controller_inputs:extend(inputs.memory[s].read_val)
                    input_size_L = input_size_L + msz * opt.num_read
                end
                x = nn.JoinTable(2)(controller_inputs)
            end
        else
            x = outputs.controller[L-1].lstm_h
            if dropout > 0 then x = nn.Dropout(dropout)(x) end
            input_size_L = rsz
        end

        local next_h, next_c = LSTM(input_size_L, rsz, x, prev_h, prev_c)
        outputs.controller[L] = Om{{lstm_c=next_c}, {lstm_h=next_h}}
    end

    local top_h = outputs.controller[n].lstm_h
    if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

    -- batch compute all controller outputs for speed
    local transfer_tbl = Om()
    local lin_nn = nn.Identity()
    local tanh_nn = nn.Tanh()
    local sig_nn = nn.Sigmoid()
    local softplus_nn = nn.SoftPlus()
    local csz = 2
    transfer_tbl[lin_nn] = Om()
    transfer_tbl[tanh_nn] = Om()
    transfer_tbl[sig_nn] = Om()

    for s=1, m do
        transfer_tbl[sig_nn][s] = Om{}
        transfer_tbl[lin_nn][s] = Om{
            {key_preread_t=listfill(ksz)},
            {key_prewrite_t=ksz},
            {act_read=listfill(ksz)},
            {act_write=ksz}
        }
        transfer_tbl[lin_nn][s].gate_read = listfill(1)
        transfer_tbl[lin_nn][s].gate_write = 1
        transfer_tbl[tanh_nn][s] = transfer_tbl[tanh_nn][s] or Om()
        transfer_tbl[tanh_nn][s].write_vec = msz
        if opt.weight_method == 'SoftMax' then
            transfer_tbl[lin_nn][s].temperature = listfill(1)
        end
        transfer_tbl[sig_nn][s].write_strength_t = 1
        transfer_tbl[sig_nn][s].gate_act_read = listfill(1)
        transfer_tbl[lin_nn][s].gate_act_write = 1
        transfer_tbl[sig_nn][s].read_val2pred_coef = listfill(1)

    end
    transfer_tbl[lin_nn].pred = vsz

    local total_size = st.get_total_size(transfer_tbl)
    local controller_outputs = nn.Linear(rsz, total_size)(top_h)
    local _splits = st.split_transfer(controller_outputs, 2, transfer_tbl)
    local splits = Om()

    for s=1, m do
        -- merge all parts in the same level, so we don't have to
        -- need to know whether they are linear or tanh or sigmoid
        -- when referring to them
        splits[s] = Om()
        splits[s]:update(_splits[lin_nn][s] or {})
        	:update(_splits[tanh_nn][s] or {})
        	:update(_splits[sig_nn][s] or {})
        local sp = splits[s]
        local in_ = inputs.memory[s]

        if opt.bias_gate_write ~= 0 then
            sp.gate_write = nn.Sigmoid()(
            	nn.AddConstant(opt.bias_gate_write)(sp.gate_write))
        else
            sp.gate_write = nn.Sigmoid()(sp.gate_write)
        end

        if opt.bias_gate_read ~= 0 then
            sp.gate_read[1] = nn.Sigmoid()(
            	nn.AddConstant(opt.bias_gate_read)(sp.gate_read[1]))
        else
            sp.gate_read[1] = nn.Sigmoid()(sp.gate_read[1])
        end
        if opt.bias_gate_act_write ~= 0 then
            sp.gate_act_write = nn.Sigmoid()(
            	nn.AddConstant(opt.bias_gate_act_write)(sp.gate_act_write))
        else
            sp.gate_act_write = nn.Sigmoid()(sp.gate_act_write)
        end
        if opt.bias_gate_act_read ~= 0 then
            sp.gate_act_read[1] = nn.Sigmoid()(
            	nn.AddConstant(opt.bias_gate_act_read)(sp.gate_act_read[1]))
        else
            sp.gate_act_read[1] = nn.Sigmoid()(sp.gate_act_read[1])
        end

        local function norm (m)
            return nn.Sqrt()(nn.Sum(2)(nn.Square()(nn.View(-1, 2)(m))))
        end

        -- compute key_read and act_read for each read head
        local key_read_t = List()
        local act_read_t = List()
        local turn_read_t = List()
        for k = 1, opt.num_read do

            local mix_extra_read = {}

            mix_extra_read.act_tm1 = in_.act_read[k]
            mix_extra_read.gate_act = sp.gate_act_read[k]

            if opt.bound_act then
                -- compute action <= action / (1 + \|action\|^p)^(1/p)
                sp.act_read[k] = nn.Normalize(opt.bound_act_exp, 1)(sp.act_read[k])
            end
            key_read_t[k], act_read_t[k] = lantm.mix_keys(sp.key_preread_t[k],
                                                    in_.key_read[k],
                                                    sp.gate_read[k],
                                                    sp.act_read[k],
                                                    opt,
                                                    mix_extra_read)
        end

        -- compute key_write and act_write
        local mix_extra_write = {}
        local turn_write_t
        mix_extra_write.act_tm1 = in_.act_write
        mix_extra_write.gate_act = sp.gate_act_write

        if opt.bound_act then
            -- compute action <= action / (1 + \|action\|^p)^(1/p)
            sp.act_write = nn.Normalize(opt.bound_act_exp, 1)(sp.act_write)
        end
        local key_write_t, act_write_t = lantm.mix_keys(sp.key_prewrite_t,
                                                in_.key_write,
                                                sp.gate_write,
                                                sp.act_write,
                                                opt,
                                                mix_extra_write)
        -- compute memory outputs
        local extra = {}
        if opt.weight_method == 'SoftMax' then
            extra.temperature = sp.temperature
        end
        extra.write_strength_t = sp.write_strength_t
        extra.mem_strength_tm1 = in_.mem_strength
        local mem_outputs = lantm.mem_step(
        	in_.mem, in_.keys_, key_write_t, key_read_t, sp.write_vec, opt, extra)

        local mem_t = mem_outputs.mem_t or error()
        local keys_t = mem_outputs.keys_t or error()
        local read_t = mem_outputs.read_t or error()
        local read_weights = mem_outputs.read_weights_t or error()
        local read_val = read_t

        outputs.memory[s] = Om{
            {mem=mem_t}, {keys_=keys_t}, {key_read=key_read_t}, {key_write=key_write_t},
            {read_val=read_val}
        }
        outputs.memory[s].mem_strength = mem_outputs.mem_strength_t
        outputs.memory[s].act_read = act_read_t
        outputs.memory[s].act_write = act_write_t
    end

    -- currently only routes the last read value (when there are many mem modules) to the output
    if m > 0 then
        outputs.pred = nn.CAddTable(){_splits[lin_nn].pred,
                                nn.CMulTable(){
                                     nn.Linear(msz, vsz)(outputs.memory[m].read_val),
                                    nn.ExpandAs(){_splits[sig_nn][m].read_val2pred_coef[1],
                                                _splits[lin_nn].pred}
                                }
                        }:annotate{name='pred'}
    else
        outputs.pred = _splits[lin_nn].pred:annotate{name='pred'}
    end
    nngraph.annotateNodes()
    return inputs, outputs
end

function lantm.model(opt, kwargs)
    kwargs = kwargs or {}
    -- enc_dict should be an (instantiated) nn.LookupTable
    local enc_dict = kwargs.enc_dict
    local criterion = kwargs.criterion
    local inputs, outputs = lantm._model(opt, kwargs)
    if enc_dict then
        -- inp will be the index of the input char
        inputs.inp = nn.Identity()():annotate{name='inp'}
        -- x is what actually gets fed into the lstm
        local x = enc_dict(inputs.inp)
        -- this essentially replace inputs.x with the local x,
        -- since the former is just nn.Identity
        inputs.x:add(x, true)
        inputs:set('x', nil)
    end
    if criterion then
        inputs.y = nn.Identity()():annotate{name='y'}
        outputs.loss = criterion{outputs.pred, inputs.y}:annotate{name='loss'}
    end
    nngraph.annotateNodes()
    local mod = nn.kModule(inputs, outputs)
    return mod
end

function lantm.init_states(opt, kwargs)
    kwargs = kwargs or {}
    local random_mag = kwargs.random_mag
    local init_states = Om({{controller=Om()}, {memory=Om()}})
    local Z = utils.gpuzeros(opt)
    if random_mag then
        function Z(...)
            return utils.gpuzeros(opt)(...):uniform(-random_mag, random_mag)
        end
    end
    local ml = opt.init_mem_length
    local nr = opt.num_read
    local bs = opt.batch_size
    for i = 1, opt.num_lstm_layers do
        init_states.controller[i] = Om()
    	local d = init_states.controller[i]
        d.lstm_c = Z(bs, opt.rnn_size)
        d.lstm_h = Z(bs, opt.rnn_size)
    end
    for i = 1, opt.num_memory_modules do
        init_states.memory[i] = Om()
        local d = init_states.memory[i]
        d.mem = listfill(ml, Z(bs, opt.mem_width))
        d.keys_ = listfill(ml, Z(bs, opt.key_dim))
        d.key_read = listfill(nr, Z(bs, opt.key_dim))
        d.key_write = Z(bs, opt.key_dim)
        d.read_val = listfill(nr, Z(bs, opt.mem_width))
        d.mem_strength = listfill(ml, Z(bs, 1))

        d.act_read = listfill(nr, Z(bs, opt.key_dim))
        d.act_write = Z(bs, opt.key_dim)
    end
    return init_states
end

function lantm.learnable_init_states(opt, kwargs)
    kwargs = kwargs or {}
    local rsz = kwargs.rnn_size or opt.rnn_size
    local id = nn.Identity()
    local transfers = Om{[id]=Om{{controller=Om()}, {memory=Om()}}}

    local cont_outs_sz = transfers[id].controller
    for i=1,opt.num_lstm_layers do
        cont_outs_sz[i] = Om{{lstm_c=rsz}, {lstm_h=rsz}}
    end

    local mem_outs_sz = transfers[id].memory
    for i=1,opt.num_memory_modules do
        mem_outs_sz[i] = Om{
            {read_val=listfill(opt.num_read, opt.mem_width)},
        }
        mem_outs_sz[i]:update(Om{
            {mem=listfill(opt.init_mem_length, opt.mem_width)},
            {keys_=listfill(opt.init_mem_length, opt.key_dim)}
        })
        mem_outs_sz[i].mem_strength = listfill(opt.init_mem_length, 1)
        mem_outs_sz[i].key_read = listfill(opt.num_read, opt.key_dim)
        mem_outs_sz[i].key_write = opt.key_dim
        mem_outs_sz[i].act_read = listfill(opt.num_read, opt.key_dim)
        mem_outs_sz[i].act_write = opt.key_dim
    end

    local in_ = nn.Identity()()
    local lm = nn.Linear(1, st.get_total_size(transfers))(in_)
    local splits = st.split_transfer(lm, 2, transfers)

    -- modifying splits into the output_map of a kModule
    local mem_outs = splits[id].memory
    local function convert2mem(m)
        return nn.Tanh()(m)
    end
    local function convert2memst(m)
        if opt.bias_init_mem_strength == 0 then
            return nn.Sigmoid()(m)
        else
            return nn.Sigmoid()(nn.AddConstant(opt.bias_init_mem_strength)(m))
        end
    end
    local function bound_norm(x)
        return nn.Normalize(opt.bound_act_exp, 1)(x)
    end
    for i=1,opt.num_memory_modules do
        local mo = mem_outs[i]
        mo.mem = mo.mem and mo.mem:map(convert2mem)
        mo.read_val = mo.read_val:map(convert2mem)

        mo.mem_strength = mo.mem_strength and mo.mem_strength:map(convert2memst)
        if opt.bound_act then
            mo.act_read = mo.act_read:map(bound_norm)
            mo.act_write = bound_norm(mo.act_write)
        end

    end
    nngraph.annotateNodes()
    return nn.kModule(List{in_}, splits[id])
end

return lantm
