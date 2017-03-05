require 'torch'
require 'nn'
require 'nngraph'
require 'paths'
require 'pl'
paths.dofile('../kwrap/kModule.lua')
paths.dofile('../layers/InnerProduct.lua')
paths.dofile('../layers/Narrow.lua')
paths.dofile('../layers/MixtureTable.lua')
paths.dofile('../layers/NarrowTable.lua')
paths.dofile('../layers/CMulList.lua')
paths.dofile('../layers/Append.lua')
local utils = paths.dofile('../utils/utils.lua')
local st = paths.dofile('../kwrap/split_transfer.lua')
local Om = OrderedMap
local pp = paths.dofile('../utils/pp.lua')
local pp_om = pp.pp_om
local listfill = utils.listfill

local ram = {}

function ram.mem_step(mem_tm1, keys_tm1, key_write_t, key_read_t, write_vec_t, opt, extra)
    -- mem_tm1 is a table of size mem_length, with elements tensors of size (batch_size, mem_width)
    -- keys_tm1 has size (batch_size, mem_length, key_dim)
    -- key_write_t and key_read_t have size (batch_size, key_dim)
    -- write_vec_t has size (batch_size, mem_width)

    -- read_weights_t has size (batch_size, mem_length)
    local read_weights_t = ram.get_read_weights(keys_tm1, key_read_t, opt, extra)
    local read_t
    -- if ram.mem_step is called through controller, then key_read_t should be a list
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
    if opt.tape then
        for i = 1, opt.num_read do
            ret.forward_keys = ret.forward_keys or List()
            ret.forward_keys:append(
                nn.MixtureTable(){
                    nn.Narrow(2, 1, -1)(read_weights_t[i]),
                    nn.NarrowTable(2, -1)(keys_tm1)
                }
            )
            ret.backward_keys = ret.backward_keys or List()
            ret.backward_keys:append(
                nn.MixtureTable(){
                    nn.Narrow(2, 2, -1)(read_weights_t[i]),
                    nn.NarrowTable(1, -1)(keys_tm1)
                }
            )
        end
    end

    return ret
end
function ram.get_read_weights(keys_tm1, key_read_t, opt, extra)
    if types.is_type(key_read_t, List) then
        local rw = List()
        for i=1,opt.num_read do
            local extra_
            if extra then
                extra_ = tablex.copy(extra)
                extra_.doubt_read = extra_.doubt_read and extra_.doubt_read[i]
            else
                extra_ = nil
            end
            rw:append(ram._get_read_weights(keys_tm1, key_read_t[i], opt, extra_))
        end
        return rw
    else
        return ram._get_read_weights(keys_tm1, key_read_t, opt, extra)
    end
end
function ram._get_read_weights(keys_tm1, key_read_t, opt, extra)
    -- scores has dimension (batch_size, mem_length)
    local scores = nn.InnerProduct(){key_read_t, keys_tm1}
    -- read_weights has dimension (batch_size, mem_length)
    local read_weights
    if extra and extra.sharpen_coef then
        -- expect extra.sharpen_coef to have shape (batch_size, 1)
        read_weights = nn.SoftMax()(nn.CMulTable(){
                            nn.ExpandAs(){extra.sharpen_coef, scores},
                            scores})
    else
        read_weights = nn.SoftMax()(scores)
    end
    -- mem_strength_tm1 has shape (batch_size, mem_length)
    read_weights =
        nn.Normalize(1)(
            nn.AddConstant(opt.epsilon_read_weight, true)(
                nn.CMulList(){extra.mem_strength_tm1, read_weights}
                )
            )
    nngraph.annotateNodes()
    return read_weights
end
function ram._model(opt, kwargs)
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
        inputs.memory[s].read_val = listinput(opt.num_read)
        inputs.memory[s].mem_strength = nn.Identity()()
        if opt.tape then
            print('ram with tape')
            inputs.memory[s].forward_keys = listinput(opt.num_read)
            inputs.memory[s].backward_keys = listinput(opt.num_read)
        else
            print('ram without tape')
        end
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
    transfer_tbl[lin_nn] = Om()
    transfer_tbl[tanh_nn] = Om()
    transfer_tbl[sig_nn] = Om()
    transfer_tbl[softplus_nn] = Om()
    for s=1,m do
        transfer_tbl[lin_nn][s] = Om{
            {key_read=listfill(ksz)}
        }
        transfer_tbl[tanh_nn][s] = transfer_tbl[tanh_nn][s] or Om()
        transfer_tbl[tanh_nn][s].write_vec = msz
        transfer_tbl[lin_nn][s].key_write = ksz
        transfer_tbl[sig_nn][s] = transfer_tbl[sig_nn][s] or Om()
        transfer_tbl[sig_nn][s].write_strength_t = 1
        if opt.wt_sharpening then
            transfer_tbl[softplus_nn][s] = transfer_tbl[softplus_nn][s] or Om()
            transfer_tbl[softplus_nn][s].sharpen_coef = 1
        end
        local lk = 1
        if opt.tape then
            lk = lk + 2
        end
        if lk > 1 then
            transfer_tbl[lin_nn][s].gate_link = listfill(lk)
        end
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
                :update(_splits[softplus_nn][s] or {})
        local sp = splits[s]
        local in_ = inputs.memory[s]

        local mixed_key_read = List()
        if sp.gate_link then
            for i = 1, opt.num_read do
                local links = List{sp.key_read[i]}
                if opt.tape then
                    links:append(inputs.memory[s].forward_keys[i])
                    links:append(inputs.memory[s].backward_keys[i])
                end

                mixed_key_read:append(nn.MixtureTable(2){
                    nn.SoftMax()(sp.gate_link[i]), nn.Identity()(links)
                })
            end
        else
            mixed_key_read = sp.key_read
        end
        -- compute memory outputs
        local extra = {}

        extra.write_strength_t = sp.write_strength_t
        extra.mem_strength_tm1 = in_.mem_strength
        if sp.sharpen_coef then
            extra.sharpen_coef = nn.AddConstant(1)(sp.sharpen_coef)
        end
        local mem_outputs = ram.mem_step(
            in_.mem, in_.keys_, sp.key_write, mixed_key_read, sp.write_vec, opt, extra)

        local mem_t = mem_outputs.mem_t or error()
        local keys_t = mem_outputs.keys_t or error()
        local read_t = mem_outputs.read_t or error()
        local read_weights = mem_outputs.read_weights_t or error()


        outputs.memory[s] = Om{
            {mem=mem_t}, {keys_=keys_t}, {read_val=read_t}
        }
        outputs.memory[s].mem_strength = mem_outputs.mem_strength_t
        if opt.tape then
            outputs.memory[s].forward_keys = mem_outputs.forward_keys
            outputs.memory[s].backward_keys = mem_outputs.backward_keys
        end

    end

    if m > 0 then
        outputs.pred = nn.CAddTable(){_splits[lin_nn].pred,
            nn.Linear(msz * opt.num_read, vsz)(
                opt.num_read > 1 and nn.JoinTable(2)(outputs.memory[m].read_val) or
                outputs.memory[m].read_val
            )}:annotate{name='pred'}
    else
        outputs.pred = _splits[lin_nn].pred:annotate{name='pred'}
    end
    nngraph.annotateNodes()
    return inputs, outputs
end


function ram.model(opt, kwargs)
    kwargs = kwargs or {}
    -- enc_dict should be an (instantiated) nn.LookupTable
    local enc_dict = kwargs.enc_dict
    local criterion = kwargs.criterion
    local inputs, outputs = ram._model(opt, kwargs)
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

function ram.init_states(opt, kwargs)
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
        d.read_val = listfill(nr, Z(bs, opt.mem_width))
        d.mem_strength = listfill(ml, Z(bs, 1))
        local fb = List{'forward', 'backward'}
        if opt.tape then
            d.forward_keys = listfill(nr, Z(bs, opt.key_dim))
            d.backward_keys = listfill(nr, Z(bs, opt.key_dim))
        end
    end
    return init_states
end

function ram.learnable_init_states(opt, kwargs)
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
        if opt.tape then
            mem_outs_sz[i].forward_keys = listfill(opt.num_read, opt.key_dim)
            mem_outs_sz[i].backward_keys = listfill(opt.num_read, opt.key_dim)
        end
    end

    local in_ = nn.Identity()()
    local lm = nn.Linear(1, st.get_total_size(transfers))(in_)
    local splits = st.split_transfer(lm, 2, transfers)

    -- modifying splits into the output_map of a kModule
    local mem_outs = splits[id].memory
    local function convert2mem(m)
        return opt.squash_mem and nn.Tanh()(m) or m
    end
    local function convert2memst(m)
        if opt.bias_init_mem_strength == 0 then
            return nn.Sigmoid()(m)
        else
            return nn.Sigmoid()(nn.AddConstant(opt.bias_init_mem_strength)(m))
        end
    end
    for i=1,opt.num_memory_modules do
        local mo = mem_outs[i]
        mo.mem = mo.mem and mo.mem:map(convert2mem)
        mo.read_val = mo.read_val:map(convert2mem)

        mo.mem_strength = mo.mem_strength and mo.mem_strength:map(convert2memst)
    end
    nngraph.annotateNodes()
    return nn.kModule(List{in_}, splits[id])
end

return ram