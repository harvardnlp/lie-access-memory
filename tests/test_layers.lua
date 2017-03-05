require 'nn'
require 'paths'
require 'pl'
require 'totem'
require 'nngraph'
nngraph.setDebug(true)
local pp = paths.dofile('../utils/pp.lua')
local utils = paths.dofile('../utils/utils.lua')
local test = torch.TestSuite()
local tester = totem.Tester()
function test.expandas()
  paths.dofile('../layers/ExpandAs.lua')
  local net = nn.ExpandAs()
  local net2 = nn.ExpandAs()
  local A = torch.rand(2, 1, 3, 1)
  local B = torch.rand(2, 5, 3, 7)
  local step = torch.zeros(A:size())
  local eps = 1e-10
  local num_grad = torch.zeros(A:size())
  local gradOutput = torch.randn(B:size())
  local out = net:forward{A, B}

  -- forward
  tester:assertTensorEq(out, A:expandAs(B), 1e-10)

  local net_grad = net:backward({A, B}, gradOutput)[1]
  for i=1,A:nElement() do
    step:storage()[i] = eps
    step_p = net:forward{A+step, B}
    step_m = net2:forward{A-step, B}
    num_grad:storage()[i] = (step_p - step_m):dot(gradOutput) / (2 * eps)
    step:storage()[i] = 0
  end

  -- backward
  tester:assertTensorEq(num_grad, net_grad, 1e-4)
end

function test.vaddtable()
  paths.dofile('../layers/VAddTable.lua')
  -- forward/backward
  for i = 1, 10 do
      local vaddtable = nn.VAddTable()
      local A = torch.rand(3, 4)
      local BList = {}
      local _BList = {}
      for i = 1, 5 do
          local B = torch.rand(3, 4)
          table.insert(BList, B)
          table.insert(_BList, torch.totable(B))
      end
      local out = vaddtable:forward{A, BList}
      local BTensor = torch.Tensor(_BList)
      local g = torch.rand(3, 5, 4)
      local addtable = nn.CAddTable()
      local _A = A:view(A:size(1), 1, A:size(2)):expand(3, 5, 4)
      local _BTensor = BTensor:transpose(1, 2) -- (3, 5, 4)
      local correct_out = addtable:forward{_A, _BTensor}
      tester:assertTensorEq(correct_out, out, 1e-9)
      local correct_grad = addtable:backward({_A, _BTensor}, g)
      local grad = vaddtable:backward({A, BList}, g)
      tester:assertTensorEq(grad[1], correct_grad[1]:sum(2):squeeze(), 1e-9)
      for i = 1, 5 do
          -- correct_grad[2] has size (3, 5, 4)
          tester:assertTensorEq(grad[2][i], correct_grad[2][{{}, i, {}}], 1e-9)
      end
  end

  -- gradient check
  require 'nngraph'
  local A_node = nn.Identity()()
  local BTensor_node = nn.Identity()()
  local BSplit = nn.SplitTable(1)(BTensor_node)
  local vaddtable = nn.VAddTable(){A_node, BSplit}
  local net = nn.gModule({A_node, BTensor_node}, {vaddtable})
  local A = torch.rand(3, 4)
  local B = torch.rand(5, 3, 4)
  totem.nn.checkGradients(tester, net, {A, B})
end

function test.powsum()
  paths.dofile('../layers/PowSum.lua')
  -- forward/backward
  for i = 1, 10 do
      local powsum = nn.PowSum(2, 3)
      local A = torch.rand(3, 4, 5)
      local out = powsum:forward(A)
      local correct_out = torch.norm(A, 2, 3):squeeze(3):pow(2)
      tester:assertTensorEq(correct_out, out, 1e-9)
  end

  -- gradient check
  require 'nngraph'
  local A_node = nn.Identity()()
  local powsum = nn.PowSum(2, 3)(A_node)
  local net = nn.gModule({A_node}, {powsum})
  local A = torch.rand(3, 4, 5)
  totem.nn.checkGradients(tester, net, A)
end

function test.invnormalize()
  paths.dofile('../layers/InvNormalize.lua')
  local function get_net(norm, eps)
      local in_ = nn.Identity()()
      local p = torch.random() % 4
      eps = eps or 1e-10
      local out = nn.InvNormalize(p, eps)(in_)
      local net = nn.gModule({in_}, {out})
      local size = {3, 2}
      norm = norm or 1
      local a = torch.rand(unpack(size)):mul(norm)
      return net, a, p
  end

  -- compare naive forward
  for i=1,3 do
      local net, a, p = get_net(1, 1e-10)
      local naive = torch.ones(a:size()):cdiv(a)
      naive:pow(p)
      naive = naive:cdiv(torch.norm(naive, 1, 2):expandAs(naive))
      local b = net:updateOutput(a)
      tester:assertTensorEq(naive, b, 1e-5)
  end

  -- gradient check
  local net, a = get_net(1, 0)
  totem.nn.checkGradients(tester, net, a)
end

function test.cmullist()
  paths.dofile('../layers/CMulList.lua')

  local function get_tensor_net()
      local inlist = nn.Identity()()
      local split = nn.SplitTable(2)(inlist)
      local intensor = nn.Identity()()
      local cmullist = nn.CMulList(){split, intensor}
      local net = nn.gModule({inlist, intensor}, {cmullist})
      return net
  end
  -- forward
  local cmullist = nn.CMulList()
  local in1 = {}
  for i = 1, 4 do
      in1[i] = torch.rand(3, 1)
  end
  local in2 = torch.rand(3, 4)
  -- print(torch.cat(in1, 2))
  -- print(in2)
  local cat = torch.cat(in1, 2)
  local cnet = nn.CMulTable()
  tester:assertTensorEq(
      cmullist:forward{in1, in2}, cnet:forward{cat, in2})
  local g = torch.rand(3, 4)
  local g1 = cmullist:backward({in1, in2}, g)
  local g2 = cnet:backward({cat, in2}, g)
  tester:assertTensorEq(torch.cat(g1[1], 2), g2[1])
  tester:assertTensorEq(g2[2], g2[2])

  -- backward
  local net = get_tensor_net()
  local correct_net = nn.CMulTable()
  for i = 1, 5 do
      local in1 = torch.rand(3, 4)
      local in2 = torch.rand(3, 4)
      tester:assertTensorEq(net:forward{in1, in2}, correct_net:forward{in1, in2})
      local grad = torch.rand(3, 4)
      local gradIn = net:backward({in1, in2}, grad)
      local correct_gradIn = correct_net:backward({in1, in2}, grad)
      tester:assertTensorEq(gradIn[1], correct_gradIn[1])
      tester:assertTensorEq(gradIn[2], correct_gradIn[2])
  end

  -- gradient
  local net = get_tensor_net()
  local in1 = torch.rand(3, 4)
  local in2 = torch.rand(3, 4)
  totem.nn.checkGradients(tester, net, List{in1, in2})
end

function test.append()
  paths.dofile('../layers/Append.lua')
  local net = nn.Append()
  local a = {torch.rand(3, 4, 5)}
  local b = torch.rand(3, 2, 5)
  local p = torch.rand(3)
  local c = net:forward{a, b, {p}}
  tester:assert(c[1] == a[1])
  tester:assert(c[2] == b)
  tester:assert(c ~= a)
  local d = {torch.rand(3, 4, 5), torch.rand(3, 2, 5)}
  local e = net:backward({a, b, {p}}, d)
  local da, db = unpack(e)
  tester:assert(d[1] == da[1])
  tester:assert(d[2] == db)
  tester:assertTensorEq(e[3][1], torch.zeros(3), 1e-16)
  tester:assert(#e[3] == 1)
  tester:assert(#a == 1)
end

function test.crossproduct()
  paths.dofile('../layers/CrossProduct.lua')
  local net = nn.CrossProduct()
  for i = 1, 10 do
    local a = torch.randn(3)
    local b = torch.randn(3)
    tester:assertTensorEq(net:forward{a, b}, torch.cross(a, b, 1), 1e-16)
  totem.nn.checkGradients(tester, net, {a, b})
  end
  for i = 1, 10 do
    local a = torch.randn(5, 3)
    local b = torch.randn(5, 3)
    tester:assertTensorEq(net:forward{a, b}, torch.cross(a, b, 2), 1e-16)
    totem.nn.checkGradients(tester, net, {a, b})
  end
end
function test.Normalize()
  paths.dofile('../layers/Normalize.lua')
  local jac = nn.Jacobian
  local precision = 1e-5
   -- compare forward against torch implementation
   -- and check gradient
  for _,p in pairs({1,2,3,4,1.5}) do
    local ini = math.random(3,10)
    local input = torch.randn(ini)
    local module = nn.Normalize(p)
    local out = module:forward(input)
    local expected = torch.div(input,input:norm(p))
    tester:assertTensorEq(out, expected, 1e-7,
                            torch.typename(module) ..' (' .. p ..') - forward err ')

    local err = jac.testJacobian(module, input, -2, 2)
    tester:assertlt(err, precision, 'error norm '..p..' on state ')
  end

  -- batch mode
  for _,p in pairs({1,2,3,4,torch.uniform()*math.random(1,10)}) do
    local ini = math.random(3,5)
    local inj = math.random(3,5)
    local input = torch.Tensor(inj, ini):zero()

    local module = nn.Normalize(p)

    local err = jac.testJacobian(module, input, -2, 2)
    tester:assertlt(err, precision, 'error norm '..p..' on state ')
  end

  -- test IO correctness
  local ini = math.random(3,5)
  local inj = math.random(3,5)
  local input = torch.Tensor(inj, ini):zero()

  local module = nn.Normalize(2)

  local ferr, berr = jac.testIO(module,input, 0.1, 2)
  tester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
  tester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)

  -- test large eps
  for _,p in pairs({1,2,3,4,1.5}) do
    local ini = math.random(3,10)
    local input = torch.randn(ini)
    local module = nn.Normalize(p, 1)
    local err = jac.testJacobian(module, input, -2, 2)
    tester:assertlt(err, precision, 'error norm '..p..' on state ')
  end

  -- batch mode
  for _,p in pairs({1,2,3,4,torch.uniform()*math.random(1,10)}) do
    local ini = math.random(3,5)
    local inj = math.random(3,5)
    local input = torch.Tensor(inj, ini):zero()

    local module = nn.Normalize(p, 1)

    local err = jac.testJacobian(module, input, -2, 2)
    tester:assertlt(err, precision, 'error norm '..p..' on state ')
  end
end

function test.cos()
  paths.dofile('../layers/Cos.lua')
  for i = 1, 10 do
    -- forward
    local cos = nn.Cos()
    local in_ = torch.rand(10, 13) * 2 * math.pi
    tester:assertTensorEq(cos:forward(in_), torch.cos(in_), 1e-9)
    -- backward
    totem.nn.checkGradients(tester, cos, in_)
  end
end

function test.sin()
  paths.dofile('../layers/Sin.lua')
  for i = 1, 10 do
    -- forward
    local sin = nn.Sin()
    local in_ = torch.rand(10, 13) * 2 * math.pi
    tester:assertTensorEq(sin:forward(in_), torch.sin(in_), 1e-9)
    -- backward
    totem.nn.checkGradients(tester, sin, in_)
  end
end

function test.innerproduct()
  paths.dofile('../layers/InnerProduct.lua')
  -- forward/backward
  for i = 1, 100 do
      local innerproduct = nn.InnerProduct()
      local A = torch.rand(3, 4)
      local BList = {}
      local _BList = {}
      for i = 1, 5 do
          local B = torch.rand(3, 4)
          table.insert(BList, B)
          table.insert(_BList, torch.totable(B))
      end
      local out = innerproduct:forward{A, BList}
      local BTensor = torch.Tensor(_BList)
      -- (3, 1, 4) x (3, 4, 5)
      -- local correct_out = torch.bmm(A:view(A:size(1), 1, A:size(2)),
      --     BTensor:transpose(1, 2):transpose(2, 3)
      -- ):view(A:size(1), #BList)
      local g = torch.rand(3, 5)
      local mm = nn.MM()
      local _A = A:view(A:size(1), 1, A:size(2))
      local _BTensor = BTensor:transpose(1, 2):transpose(2, 3)
      local correct_out = mm:forward{_A, _BTensor}:view(A:size(1), #BList)
      tester:assertTensorEq(correct_out, out, 1e-9)
      local correct_grad = mm:backward({_A, _BTensor}, g:view(3, 1, 5))
      local grad = innerproduct:backward({A, BList}, g)
      tester:assertTensorEq(grad[1], correct_grad[1]:view(3, 4), 1e-9)
      for i = 1, 5 do
          -- correct_grad[2] has size (3, 4, 5)
          tester:assertTensorEq(grad[2][i], correct_grad[2][{{}, {}, i}], 1e-9)
      end
  end

  -- grad check
  require 'nngraph'
  local A_node = nn.Identity()()
  local BTensor_node = nn.Identity()()
  local BSplit = nn.SplitTable(1)(BTensor_node)
  local innerproduct = nn.InnerProduct(){A_node, BSplit}
  local net = nn.gModule({A_node, BTensor_node}, {innerproduct})
  local A = torch.rand(3, 4)
  local B = torch.rand(5, 3, 4)
  totem.nn.checkGradients(tester, net, {A, B})
end

tester:add(test):run()
