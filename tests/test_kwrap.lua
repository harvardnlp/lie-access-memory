require 'paths'
require 'pl'
require 'nngraph'
require 'totem'

paths.dofile('../kwrap/init.lua')
local st = paths.dofile('../kwrap/split_transfer.lua')
local Om = OrderedMap

local test = torch.TestSuite()
local tester = totem.Tester()
local kWrap = nn.kWrap
function test.kwrap_submap()
    tester:assert(kWrap():submap(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6, c=1, d=1}, 'keys'))
    tester:assert(kWrap():submap(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6, c=1}, 'keys'))
    tester:assert(not kWrap():submap(Om{{a=1}, {b=2}, {c=3}}, 1, 'keys'))
    tester:assert(not kWrap():submap(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6}, 'keys'))
    tester:assert(not kWrap():submap(Om{{b=Om{bb=1}}, {a=Om{aa=1}}, {c=Om{cc='hello'}}},
                                    {a=5, b=6, c=1, d=1}, 'keys'))
    tester:assert(kWrap():submap(Om{{b=Om{bb=1}}, {a=Om{aa=1}}, {c=Om{cc='hello'}}},
                                    {a=Om{aa=5}, b=Om{bb=6}, c=Om{cc=1}}, 'keys'))
end
function test.kwrap_split_input()
    local kmod = kWrap(nil, Om{{a=1}, {b=2}, {c=3}})
    tester:asserteq(kmod:merge_inputs({a=5, b=6, c=1}), List{5, 6, 1})
    local kmod = kWrap(nil, Om{{a=Om{aa=1}}, {b=Om{bb=1}}, {c=Om{cc='hello'}}})
    tester:asserteq(kmod:merge_inputs({a={aa=5}, b={bb=6}, c={cc=1}}), List{5, 6, 1})
    kmod = kWrap(nil, Om{{b=Om{bb=1}}, {a=Om{aa=1}}, {c=Om{cc='hello'}}})
    tester:asserteq(kmod:merge_inputs({a=Om{aa=5}, b=Om{bb=6}, c=Om{cc=1}}), List{6, 5, 1})
    kmod = kWrap(nil, Om{{b=Om{bb=1}}, {c=Om{cc='hello'}}, {a=Om{aa=1}}})
    tester:asserteq(kmod:merge_inputs({a={aa=5}, b={bb=6}, c={cc=1}}), List{6, 1, 5})
    
    -- extraneous input
    kmod = kWrap(nil, Om{{a='shit'}})
    tester:asserteq(kmod:merge_inputs({a=1, b=2, c=3}), List{1})
    kmod = kWrap(nil, Om{{aa='h'}, {a1='foo'}, {a2='bar'}})
    tester:asserteq(kmod:merge_inputs({aa=1, a1=2, a2=0, c=8}), List{1, 2, 0})
    
    -- complex
    kmod = kWrap(nil, Om{{a=List{'aa', 'a1', 'a2'}}, {b='bb'}, {c='cc'}})
    tester:asserteq(kmod:merge_inputs({a={0, -1, 'hell'}, b='owo', c='xyz'}), 
                    List{0, -1, 'hell', 'owo', 'xyz'})
    
    kmod = kWrap(nil, Om{{a=List{'aa', 'a1', 'a2'}},
                                                {b=Om{{b1='b11'}, 
                                                        {b2=List{'bb', 'bbb', 'bbbb'}}}},
                                                {c='cc'}})
    tester:asserteq(kmod:merge_inputs({a={0, -1, 'hell'}, 
                                        b={b1=9,
                                            b2={-2, -3, -9}}, 
                                        c=111}), 
                    List{0, -1, 'hell', 9, -2, -3, -9, 111})
end
function test.kwrap_collect_output()
    local kmod = kWrap(nn.JoinTable(1), List{'aa', 'a1', 'a2'}, Om{uu='stuff'})
    tester:asserteq(kmod:collect_outputs{1}, Om{uu=1})
    
    local kmod = kWrap(nn.JoinTable(1), List{'aa', 'a1', 'a2'}, Om{uu='stuff'})
    tester:assertError(function () kmod:collect_outputs{1, 2, 3, 4} end)
    
    kmod = kWrap(nil,
                    Om{{a=List{'aa', 'a1', 'a2'}}, {b='bb'}, {c='cc'}},
                    Om{{x='xx'}, {y=List{'y1', 'y2'}}})
    tester:asserteq(kmod:collect_outputs({1, 2, 3}), 
                    Om{{x=1}, {y=List{2, 3}}}
                    )
    kmod = kWrap(nil,
                    Om{{a=List{'aa', 'a1', 'a2'}}, {b='bb'}, {c='cc'}},
                    Om{{x='xx'}, 
                       {y=List{'y1', 'y2'}},
                       {z=Om{{z1='zz'}, 
                             {z2=Om{u=List{0, 1, 2}}}
                            }
                       }
                      }
                   )
    tester:asserteq(kmod:collect_outputs({1, 2, 3, 4, 5, 6, 7}), 
                    Om{{x=1}, 
                       {y=List{2, 3}},
                       {z=Om{{z1=4},
                             {z2=Om{u=List{5, 6, 7}}}
                            }
                       }
                      }
                   )
end

function test.kwrap_forward()
    local kmod = kWrap(nn.JoinTable(1),
                        Om{{aa=0}, {a1=0}, {a2=0}},
                        Om{uu=0}
                        )
    local m = Om{aa=torch.rand(2, 3), a1=torch.rand(5, 3), a2=torch.rand(5, 3), d=3}
    local fd = kmod:forward(m)
    tester:assertTensorEq(fd.uu, torch.cat({m.aa, m.a1, m.a2}, 1), 1e-9)
    tester:asserteq(fd:keys(), List{'uu'})
    
    -- split_transfer
    local id_nn = nn.Identity()
    local tanh_nn = nn.Tanh()
    local transfer_tbl = Om{
                    {[id_nn]=List{3, 5, 1}},
                    {[tanh_nn]=Om{{u=Om{
                                    {hello=4}, 
                                    {world=9}
                                }}, 
                                {v=6}}
                    }
    }
    local in_ = nn.Identity()()
    local outs = st.split_transfer(in_, 2, transfer_tbl)
    local net = nn.gModule({in_}, st.flatten_values(outs))
    
    local total_size = st.get_total_size(transfer_tbl)
    local input = torch.rand(2, total_size)
    local kmod = kWrap(net, Om{all=0}, transfer_tbl)
    local out_om = Om{{[id_nn]=List{input:narrow(2, 1, 3), input:narrow(2, 4, 5), input:narrow(2, 9, 1)}},
                      {[tanh_nn]=Om{{u=Om{{hello=nn.Tanh():forward(input:narrow(2, 10, 4))},
                                          {world=nn.Tanh():forward(input:narrow(2, 14, 9))}}
                                    },
                                    {v=nn.Tanh():forward(input:narrow(2, 23, 6))}
                                  }
                      }}
    local compare = function(a, b) return torch.max(torch.abs(torch.add(a, -b))) < 1e-8 end
    local net_out = kmod:forward(Om{all=input})
    tester:assert(kmod:submap(out_om, net_out, compare))
    tester:assert(kmod:submap(net_out, out_om, compare))
    local net_out = kmod:forward(List{input})
    tester:assert(kmod:submap(out_om, net_out, compare))
    tester:assert(kmod:submap(net_out, out_om, compare))
end
function test.kwrap_backward()
    local id_nn = nn.Identity()
    local tanh_nn = nn.Tanh()
    local transfer_tbl = Om{
                    {[id_nn]=List{3, 5, 1}},
                    {[tanh_nn]=Om{{u=Om{
                                    {hello=4}, 
                                    {world=9}
                                }}, 
                                {v=6}}
                    }
    }
    local in_ = nn.Identity()()
    local outs = st.split_transfer(in_, 2, transfer_tbl)
    local net = nn.gModule({in_}, st.flatten_values(outs))
    
    local total_size = st.get_total_size(transfer_tbl)
    local input = torch.rand(2, total_size)
    local kmod = kWrap(net, Om{all=0}, transfer_tbl)
    local out_om = Om{{[id_nn]=List{input:narrow(2, 1, 3), input:narrow(2, 4, 5), input:narrow(2, 9, 1)}},
                      {[tanh_nn]=Om{{u=Om{{hello=nn.Tanh():forward(input:narrow(2, 10, 4))},
                                          {world=nn.Tanh():forward(input:narrow(2, 14, 9))}}
                                    },
                                    {v=nn.Tanh():forward(input:narrow(2, 23, 6))}
                                  }
                      }}
    local compare = function(a, b) return torch.max(torch.abs(torch.add(a, -b))) < 1e-8 end
    local net_out = kmod:forward(Om{all=input})
    tester:assert(kmod:submap(out_om, net_out, compare))
    tester:assert(kmod:submap(net_out, out_om, compare))
    local gradin = kmod:backward(List{input}, net_out)
    
    local merged_outs = torch.cat(kmod:merge(kmod.output_map, net_out), 2)
    local tanh_part = merged_outs:narrow(2, 10, 19)

    local tanh_back = (torch.ones(2, 19) - torch.pow(tanh_part, 2)):cmul(tanh_part)
    local gradin_c = torch.cat({input:narrow(2, 1, 9),
                                tanh_back}, 2)
    tester:assertTensorEq(gradin_c, gradin.all, 1e-9)
end


function test.st_split_sizes_constants()
    tester:asserteq(st.get_split_sizes(Om
            {
                {aa=
                    Map{
                        a=Om{
                            {u=3},
                            {v=-1}
                        },
                        b=2
                    }
                },
                {bb=
                    Map{c=5,u=9}
                },
                {cc=Map{h=3}
                }}
        ),
                    OrderedMap({{aa=4}, {bb=14}, {cc=3}}))

end
function test.st_constant_sizes()
    local id_nn = nn.Identity()
    local tanh_nn = nn.Tanh()
    local transfer_tbl = OrderedMap{
        {[id_nn]=List{3, 5, 1}},
        {[tanh_nn]=Map{
                OrderedMap{{hello=4}, {world=9}}, 
                ['!!']=6}
        }
    }
    local in_ = nn.Identity()()
    local outs = st.split_transfer(in_, 2, transfer_tbl)
    
    local net = nn.gModule({in_}, st.flatten_values(outs))
    local total_size = st.get_total_size(transfer_tbl)
    local input = torch.rand(2, total_size)
    local net_output = net:forward(input)
    tester:assertTensorEq(input[{{}, {1, 3}}], net_output[1], 1e-10)
    tester:assertTensorEq(input[{{}, {4, 8}}], net_output[2], 1e-10)
    tester:assertTensorEq(input[{{}, {9, 9}}], net_output[3], 1e-10)
    tester:assertTensorEq(tanh_nn:forward(input[{{}, {10, 13}}]), net_output[4], 1e-10)
    tester:assertTensorEq(tanh_nn:forward(input[{{}, {14, 22}}]), net_output[5], 1e-10)
    tester:assertTensorEq(tanh_nn:forward(input[{{}, {23, 28}}]), net_output[6], 1e-10)
end


function test.kmodule_forward()
    -- split_transfer
    local id_nn = nn.Identity()
    local tanh_nn = nn.Tanh()
    local transfer_tbl = Om{
                    {[id_nn]=List{3, 5, 1}},
                    {[tanh_nn]=Om{{u=Om{
                                    {hello=4}, 
                                    {world=9}
                                }}, 
                                {v=6}}
                    }
    }
    local in_ = nn.Identity()()
    local outs = st.split_transfer(in_, 2, transfer_tbl)
    local net = nn.kModule(List{in_}, outs)
    local total_size = st.get_total_size(transfer_tbl)
    local input = torch.rand(2, total_size)
    local out_om = Om{{[id_nn]=List{input:narrow(2, 1, 3), input:narrow(2, 4, 5), input:narrow(2, 9, 1)}},
                      {[tanh_nn]=Om{
                                {u=Om{
                                    {hello=nn.Tanh():forward(input:narrow(2, 10, 4))},
                                    {world=nn.Tanh():forward(input:narrow(2, 14, 9))}
                                    }
                                },
                                {v=nn.Tanh():forward(input:narrow(2, 23, 6))}
                          }
                      }}
    local net_out = net:forward(List{input})
    local compare = function(a, b) return torch.max(torch.abs(torch.add(a, -b))) < 1e-8 end
    tester:assert(net:submap(out_om, net_out, compare))
    tester:assert(net:submap(net_out, out_om, compare))
    local gradin = net:backward(List{input}, net_out)
    local merged_outs = torch.cat(net:merge(net.output_map, net_out), 2)
    local tanh_part = merged_outs:narrow(2, 10, 19)
    
    local tanh_back = (torch.ones(2, 19) - torch.pow(tanh_part, 2)):cmul(tanh_part)
    local gradin_c = torch.cat({input:narrow(2, 1, 9),
                                tanh_back}, 2)
    tester:assertTensorEq(gradin_c, gradin[1], 1e-9)
end

tester:add(test):run()
