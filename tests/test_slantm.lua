require 'nn'
require 'paths'
require 'pl'
require 'totem'
local pp = paths.dofile('../utils/pp.lua')
local utils = paths.dofile('../utils/utils.lua')
local model = paths.dofile('../models/slantm.lua')
local test = torch.TestSuite()
local tester = totem.Tester()
local function randS1(...)
    return nn.Normalize(2):forward(torch.randn(...))
end
local function rodrigues(k, cos, sin, v)
    local line1 = torch.cmul(v, cos:expandAs(v))
    local line2 = torch.cmul(torch.cross(k, v, 2), sin:expandAs(v))
    local line3 = torch.cmul(1 - cos:expandAs(v),
                    torch.cmul(k,
                    nn.DotProduct():forward{k, v}:view(-1, 1):expandAs(k))
                )
    return line1 + line2 + line3
end
local function mixture(k, k_tm1, gate)
	local kpart = torch.cmul(gate:expandAs(k), k)
	local ktm1part = torch.cmul((1 - gate):expandAs(k), k_tm1)
	return nn.Normalize(2):forward(kpart + ktm1part)
end
function test.mix_keys()
	-- mix keys
	local ins = {}
	-- {'key_t', 'key_tm1', 'gate', 'rot_axis', 'rot_trig'}
	for k = 1, 5 do
	    ins[k] = nn.Identity()()
	end
	local out = {model.mix_keys(unpack(ins))}
	local net = nn.gModule(ins, out)

	-- no mixture; gate = 1
	for j = 1, 10 do
		local key_t = randS1(4, 3)
		local key_tm1 = randS1(4, 3)
		local gate = torch.ones(4, 1)
		local rot_axis = randS1(4, 3)
		local rot_trig = randS1(4, 2)
		local outputs = net:forward{
			key_t,
			key_tm1,
			gate,
			rot_axis,
			rot_trig}
		local vrot = rodrigues(rot_axis, rot_trig[{{}, {1, 1}}], rot_trig[{{}, {2, 2}}], key_t)
		tester:assertTensorEq(outputs[1], vrot, 1e-7)
		for i = 1, 3 do
			tester:assertTensorEq(outputs[i]:norm(2, 2), torch.ones(4, 1), 1e-7)
		end
	end
	-- mixture
	for j = 1, 10 do
		local key_t = randS1(4, 3)
		local key_tm1 = randS1(4, 3)
		local gate = torch.rand(4, 1)
		local rot_axis = randS1(4, 3)
		local rot_trig = randS1(4, 2)
		local outputs = net:forward{
			key_t,
			key_tm1,
			gate,
			rot_axis,
			rot_trig}
		local mixed = mixture(key_t, key_tm1, gate)
		local vrot = rodrigues(rot_axis, rot_trig[{{}, {1, 1}}], rot_trig[{{}, {2, 2}}], mixed)
		tester:assertTensorEq(outputs[1], vrot, 1e-7)
		for i = 1, 3 do
			tester:assertTensorEq(outputs[i]:norm(2, 2), torch.ones(4, 1), 1e-7)
		end
	end
end

tester:add(test):run()