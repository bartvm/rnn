local Distribute = torch.class('nn.Distribute', 'nn.Sequencer')

local function distribute(inputs)
    local result = {}
    for i = 1, #inputs[1] do
      result[i] = {inputs[1][i]}
      for j = 2, #inputs do
        result[i][j] = inputs[j]
      end
    end
    return result
end

local function merge(gradInputs)
    local result = {{}}
    for i = 1, #gradInputs do
        result[1][i] = gradInputs[i][1]
        for j = 2, #gradInputs[i] do
            if result[j] then
              result[j] = nn.rnn.recursiveAdd(result[j], gradInputs[i][j])
            else
              result[j] = nn.rnn.recursiveClone(gradInputs[i][j])
            end
        end
    end
    return result
end

function Distribute:updateOutput(inputTable)
    return nn.Sequencer.updateOutput(self, distribute(inputTable))
end

function Distribute:updateGradInput(inputTable, gradOutputTable)
    gradInput = nn.Sequencer.updateGradInput(self, distribute(inputTable), gradOutputTable)
    self.gradInput = merge(gradInput)
    return self.gradInput
end

function Distribute:accGradParameters(inputTable, gradOutputTable, scale)
  return nn.Sequencer.accGradParameters(self, distribute(inputTable), gradOutputTable, scale)
end

function Distribute:accUpdateGradParameters(inputTable, gradOutputTable, scale)
  return nn.Sequencer.accUpdateGradParameters(self, distribute(inputTable), gradOutputTable, scale)
end
