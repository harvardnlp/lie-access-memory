require 'pl'
require 'paths'
local utils = paths.dofile('utils/utils.lua')
local Om = OrderedMap

local fb = {}

function fb.init(opt, init_states)
    local Z = utils.gpuzeros(opt)
    fb.loss_grad = utils.gpu(torch.ones(1), opt)
    fb.zero_loss_grad = Z(1)
    fb.pred_grad = Z(opt.batch_size, tablex.size(loader.idx2char))
    fb.grads = init_states(opt)
    fb.blank_input = Z(opt.batch_size):fill(loader.char2idx['#'])
    fb.batched_zeros = List{Z(opt.batch_size, 1)}
    fb.read_weights_grad = Z(opt.batch_size, 4 * opt.max_valid_seq_len)
end
function fb.makememlist(state)
    if not state.memory or #state.memory == 0 then
        return
    end
    for i, M in ipairs(state.memory) do
        M.mem = List(M.mem)
        M.keys_ = List(M.keys_)
        M.mem_strength = List(M.mem_strength)
    end
end
function fb.forward(state, opt, protos, clones, kwargs)
    kwargs = kwargs or {}
    local calc_coarse
    if kwargs.calc_coarse == nil then
        calc_coarse = true
    else
        calc_coarse = kwargs.calc_coarse
    end
    -- The rnn state going into timestep t.
    local rnn_states
    rnn_states = {[0] = protos.init_states:forward(fb.batched_zeros)}
    __rnn_states = rnn_states
    local predictions = {}
    local loss = 0
    local t = 0
    local nloss = 0
    local answerphase = false
    local coarse
    if calc_coarse then
        coarse = torch.LongTensor(opt.batch_size, 1):fill(1)
    end
    local function char_at(pos)
        return loader.idx2char[state.data[{pos, 1}]]
    end
    assert(char_at(state.pos) == '^')
    -- we are guaranteed that the last char is $
    while char_at(state.pos) ~= '$' do
        local inp = answerphase and opt.blank_input_during_response and fb.blank_input or
                                            state.data[state.pos]:long()
        if not answerphase then
            answerphase = char_at(state.pos) == '#'
        end
        local y = state.data[state.pos + 1]
        rnn_states[t]:set('answerphase', answerphase)
        rnn_states[t]:set('inp', inp)
        rnn_states[t]:set('y', utils.convert2Reals(y))
        rnn_states[t+1] = clones.rnn[t]:forward(rnn_states[t])
        rnn_states[t].__rnn__ = t -- for use in backprop
        if answerphase and opt.no_write_during_response and opt.num_memory_modules > 0 then
            -- pop the added memory and keys to __write_popped__
            rnn_states[t+1].__write_popped__ = List()
            for i = 1, #rnn_states[t+1].memory do
                local M = rnn_states[t+1].memory[i]
                rnn_states[t+1].__write_popped__[i] = Om()
                local W = rnn_states[t+1].__write_popped__[i]
                W.mem = M.mem:pop()
                W.keys_ = M.keys_:pop()
                W.mem_strength = M.mem_strength:pop()
            end
        end
        if answerphase then
            -- note: criterion as an nngraph node produces a size (1,) tensor
            loss = loss + rnn_states[t+1].loss[1]
            nloss = nloss + 1
            local predp = rnn_states[t+1].pred
            local _, pred = torch.max(predp, 2)
            local predright = pred:long():eq(y:long()):long()
            state.fine = state.fine + predright:sum()
            state.fine_total = state.fine_total + predright:nElement()
            state.fine_acc = state.fine / state.fine_total
            if calc_coarse then
                predright:maskedFill(y:eq(0):byte(), 1)
                coarse:cmul(predright)
            end
        end
        state.pos = state.pos + 1
        t = t + 1
    end
    -- state.pos is $, so state.pos + 1 is ^
    state.pos = state.pos + 1
    if coarse then
        state.coarse = state.coarse + coarse:sum()
        state.coarse_total = state.coarse_total + coarse:nElement()
        state.coarse_acc = state.coarse / state.coarse_total
    end
    local cycle = false
    if state.pos > state.data:size(1) then
        state.pos = 1
        cycle = true
    end
    local ret = {
        rnn_states=rnn_states,
        loss=loss,
        nloss=nloss,
    }
    return ret, cycle
end

function fb.backward(state, rnn_states, opt, protos, clones)
    -- copy rnn_states[#rnn_states] into a new nested collection, with the tensors being
    -- the zeroed versions of the corresponding tensors in rnn_states[#rnn_states],
    -- which is taken from fb.grads whenever fb.grads has such a zeroed tensor
    -- so as to save memory and avoid extraneous allocation
    local d_state = utils.zero_nested(rnn_states[#rnn_states], opt, false, fb.grads)

    -- exposing grads for inspection and debugging
    __d_states = {}
    for t = #rnn_states-1, 0, -1 do
        local answerphase = rnn_states[t].answerphase
        d_state.loss = answerphase and fb.loss_grad or fb.zero_loss_grad
        d_state.pred = fb.pred_grad
        fb.makememlist(d_state) -- required since nngraph can't return Lists
        __d_states[t] = d_state
        if rnn_states[t+1].__write_popped__ then
            for i = 1, #d_state.memory do
                local M = d_state.memory[i]
                table.insert(M.mem, fb.grads.memory[i].mem[1])
                table.insert(M.keys_, fb.grads.memory[i].keys_[1])
                table.insert(M.mem_strength, fb.grads.memory[i].mem_strength[1])
            end
        end
        assert(rnn_states[t].__rnn__ == t)
        d_state = clones.rnn[t]:backward(rnn_states[t], d_state)
        if cutorch then
            cutorch.synchronize()
        end
    end
    d_state = protos.init_states:backward(fb.batched_zeros, d_state)
    return d_state
end

return fb
