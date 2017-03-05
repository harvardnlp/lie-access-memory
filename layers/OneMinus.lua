require 'nn'
require 'nnx'
function nn.OneMinus()
    return
    function (layer)
        return nn.Minus()(nn.AddConstant(-1)(layer))
    end
end