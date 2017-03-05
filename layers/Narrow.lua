local Narrow = nn.Narrow

function Narrow:updateOutput(input)
    self._length = self.length < 1 and input:size(self.dimension) + self.length or self.length
    local output = input:narrow(self.dimension, self.index, self._length)
    self.output = self.output:typeAs(output)
    self.output:resizeAs(output):copy(output)
    return self.output
end

function Narrow:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput:typeAs(input)
   self.gradInput:resizeAs(input):zero()
   self.gradInput:narrow(self.dimension, self.index, self._length):copy(gradOutput)
   return self.gradInput
end