local LSTMGatesX, parent
--[[LSTMGatesX(in_size, rnn_size) is just nn.Linear(in_size, 4 * rnn_size), where the (4 * rnn_size) splits into 4 parts,
resp. the forget, in, and out gates, and in_transform, the new input into the hidden state.
The only difference is that there's custom init for the bias of the part for forget gates]]
if not pcall(function()
            LSTMGatesX, parent = torch.class('nn.LSTMGatesX', 'nn.Linear')
         end) then
    LSTMGatesX = nn.LSTMGatesX
    parent = nn.Linear
end

function LSTMGatesX:__init(in_size, rnn_size)
    parent.__init(self, in_size, 4 * rnn_size)
end
function LSTMGatesX:reset(stdv, forget_bias, in_bias, out_bias)
    parent.reset(self, stdv)
    local s = self.weight:size(1)/4
    if forget_bias then
        self.bias:narrow(1, 1, s):fill(forget_bias)
    end
    if in_bias then
        self.bias:narrow(1, s + 1, s):fill(in_bias)
    end
    if out_bias then
        self.bias:narrow(1, 2 * s + 1, s):fill(out_bias)
    end

    return self
end

function LSTM(isz, rsz, x, prev_h, prev_c, kw)
    kw = kw or {}
    -- evaluate the input sums at once for efficiency
    local i2h = nn.LSTMGatesX(isz, rsz)(x)
    local h2h = nn.LSTMGatesX(rsz, rsz)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.View(4, rsz):setNumInputDims(1)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local forget_gate = nn.Sigmoid()(n1)
    local in_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    local in_ = nn.CMulTable()({in_gate, in_transform})
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        in_
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    nngraph.annotateNodes()
    return next_h, next_c
end