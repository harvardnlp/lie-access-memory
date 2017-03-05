require 'torch'
require 'nn'
require 'nngraph'
require 'pl'
require 'paths'
paths.dofile('../kwrap/kModule.lua')
local utils = paths.dofile('../utils/utils.lua')
local st = paths.dofile('../kwrap/split_transfer.lua')
local Om = OrderedMap
local pp = paths.dofile('../utils/pp.lua')
local pp_om = pp.pp_om

local lstm = {}

function lstm._model(opt, kwargs)
    kwargs = kwargs or {}
    -- convenience variables
    local function getdropout(layer, which)
        local d = opt.dropout
        if not d then return 0 end
        if type(d) == 'table' then
            return d[layer] or 0
        else
            return d
        end
    end
    local n = opt.num_lstm_layers
    local esz = opt.embedding_size
    local vsz = opt.vocab_size
    local rsz = opt.rnn_size

    local inputs = Om()
    inputs.x = nn.Identity()()
    inputs.controller = Om()
    for L = 1, n do
        inputs.controller[L] = Om()
        inputs.controller[L].lstm_c = nn.Identity()()
        inputs.controller[L].lstm_h = nn.Identity()()
    end

    local outputs = Om()
    outputs.controller = Om()

    local x, input_size_L

    for L = 1, n do
        -- cell and hidden state from previous timesteps
        local prev_c = inputs.controller[L].lstm_c
        local prev_h = inputs.controller[L].lstm_h

        -- the input to this layer
        if L == 1 then
            x = inputs.x
            input_size_L = esz
        else
            x = outputs.controller[L-1].lstm_h
            input_size_L = rsz
        end
        if getdropout(L) > 0 then x = nn.Dropout(getdropout(L))(x) end

        local next_h, next_c, extra = LSTM(input_size_L, rsz, x, prev_h, prev_c,
                                            lstmkw)
        outputs.controller[L] = Om{{lstm_c=next_c}, {lstm_h=next_h}}
    end

    local top_h = outputs.controller[n].lstm_h
    if getdropout(n+1) > 0 then top_h = nn.Dropout(getdropout(n+1))(top_h) end

    -------------------------------------------------
    -- batch compute all controller outputs for speed
    -------------------------------------------------
    local transfer_tbl = Om()
    local lin_nn = nn.Identity()
    local tanh_nn = nn.Tanh()
    local sig_nn = nn.Sigmoid()
    transfer_tbl[lin_nn] = Om()
    transfer_tbl[tanh_nn] = Om()
    transfer_tbl[sig_nn] = Om()

    -- register the name and size of the output
    transfer_tbl[lin_nn].pred = vsz

    -- compute total size of output
    local total_size = st.get_total_size(transfer_tbl)
    local controller_outputs = nn.Linear(rsz, total_size)(top_h)

    -- transfer_tbl acts as a splitter that splits controller_outputs into named outputs
    -- (and apply transfer functions to the splits).
    -- The results are stored in _splits, which is an ordered map.
    -- 2 is the dimension split (usually dimension 1 is the batch dimension).
    local _splits = st.split_transfer(controller_outputs, 2, transfer_tbl)
    -------------------------------------------------

    outputs.pred = _splits[lin_nn].pred:annotate{name='pred'}

    nngraph.annotateNodes()
    return inputs, outputs
end
function lstm.model(opt, kwargs)
    kwargs = kwargs or {}
    -- enc_dict should be an (instantiated) nn.LookupTable
    local enc_dict = kwargs.enc_dict
    local criterion = kwargs.criterion
    local inputs, outputs = lstm._model(opt, kwargs)
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

function lstm.init_states(opt, kwargs)
    kwargs = kwargs or {}
    local random_mag = kwargs.random_mag
    local init_states = Om({{controller=Om()}})
    local Z = utils.gpuzeros(opt)
    if random_mag then
        function Z(...)
            return utils.gpuzeros(opt)(...):uniform(-random_mag, random_mag)
        end
    end
    for i = 1, opt.num_lstm_layers do
        init_states.controller[i] = Om()
        init_states.controller[i].lstm_c = Z(opt.batch_size, opt.rnn_size)
        init_states.controller[i].lstm_h = Z(opt.batch_size, opt.rnn_size)
    end
    return init_states
end

function lstm.learnable_init_states(opt, kwargs)
    kwargs = kwargs or {}
    local random_mag = kwargs.random_mag
    local id = nn.Identity()
    local transfers = Om{[id]=Om{{controller=Om()}}}
    local cont_outs_sz = transfers[id].controller
    local rsz = opt.rnn_size
    for i=1,opt.num_lstm_layers do
        cont_outs_sz[i] = Om{{lstm_c=rsz}, {lstm_h=rsz}}
    end
    local in_ = nn.Identity()()
    local lm = nn.Linear(1, st.get_total_size(transfers))(in_)
    if random_mag then
        lm:reset(random_mag / math.sqrt(3))
    end
    local splits = st.split_transfer(lm, 2, transfers)
    nngraph.annotateNodes()
    return nn.kModule(List{in_}, splits[id])
end

return lstm
