require 'paths'
require 'pl'
require 'totem'
local nested = paths.dofile('../../utils/nested.lua')
local Om = OrderedMap

local test = torch.TestSuite()
local tester = totem.Tester()
function test.nested_submap()
    tester:assert(nested.submap(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6, c=1, d=1}, 'keys'))
    tester:assert(nested.submap(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6, c=1}, 'keys'))
    tester:assert(not nested.submap(Om{{a=1}, {b=2}, {c=3}}, 1, 'keys'))
    tester:assert(not nested.submap(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6}, 'keys'))
    tester:assert(not nested.submap(Om{{b=Om{bb=1}}, {a=Om{aa=1}}, {c=Om{cc='hello'}}},
                                    {a=5, b=6, c=1, d=1}, 'keys'))
    tester:assert(nested.submap(Om{{b=Om{bb=1}}, {a=Om{aa=1}}, {c=Om{cc='hello'}}},
                                    {a=Om{aa=5}, b=Om{bb=6}, c=Om{cc=1}}, 'keys'))
end
function test.nested_mapeq()
    tester:assert(not nested.mapeq(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6, c=1, d=1}, 'keys', true))
    tester:assert(not nested.mapeq(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6, c=1}, 'keys', true))
    tester:assert(nested.mapeq(Om{{a=1}, {b=2}, {c=3}}, Om{{a=5}, {b=6}, {c=1}}, 'keys', true))
    tester:assert(not nested.mapeq(Om{{a=1}, {b=2}, {c=3}}, Map{a=5, b=6, c=1, d=1}, 'keys', true))
    tester:assert(nested.mapeq(Om{{a=1}, {b=2}, {c=3}}, {a=5, b=6, c=1}, 'keys'))
    tester:assert(nested.mapeq(Om{{a=1}, {b=2}, {c=3}}, Om{{a=5}, {b=6}, {c=1}}, 'keys'))
    tester:assert(not nested.mapeq(Om{{a=1}, {b=2}, {c=3}}, 1, 'keys'))
end
function test.nested_dumploadtable()
    local function dumpload(m)
        return nested.loadfromtable(nested.dump2table(m))
    end
    local function r()
        return torch.random()
    end
    local a = Om{{[r()] = r()}, {[r()] = r()}}
    tester:assert(nested.mapeq(a, dumpload(a), 'keys', true))
    
    local a = List{3, 2, 1}
    tester:assert(nested.mapeq(a, dumpload(a), 'keys', true))
    
    local a = Om{{b=3}, {c=List{4, 3}}, {a=1}}
    tester:assert(nested.mapeq(a, dumpload(a), 'keys', true))

    local a = List{
        Om{
            {[r()] = r()},
            {[r()] = r()},
            {[r()] = List{
                    0,
                    10,
                    11
                }
            }
        },
        Om{
            {r=2},
            {s=1},
            {f=-1}
        },
        List{'asdf', 'fdsa'}
    }
        
    tester:assert(nested.mapeq(a, dumpload(a), 'keys', true))
end
function test.nested_dumploadtabletensor()
    local function dumpload(m)
        return nested.loadfromtable(nested.dump2table(m, {keep_tensor=true}))
    end
    
    local function rn()
        return torch.random()
    end
    local function r()
        local a = torch.random() % 5 + 1
        local b = torch.random() % 10 + 1
        return torch.randn(a, b)
    end
    
    local function Teq(a, b)
        if torch.isTensor(a) then
            return totem.areTensorsEq(a, b, 1e-5)
        else
            return a == b
        end
    end
    
    local a = Om{{[rn()] = r()}, {[rn()] = r()}}
    tester:assert(nested.mapeq(a, dumpload(a), Teq))
    
    local a = List{3, 2, 1}
    tester:assert(nested.mapeq(a, dumpload(a), Teq))
    
    local a = Om{{b=3}, {c=List{4, 3}}, {a=1}}
    tester:assert(nested.mapeq(a, dumpload(a), Teq))

    local a = List{
        Om{
            {[rn()] = r()},
            {[rn()] = r()},
            {[rn()] = List{
                    0,
                    10,
                    11
                }
            }
        },
        Om{
            {r=2},
            {s=1},
            {f=-1}
        },
        List{'asdf', 'fdsa'}
    }
    tester:assert(nested.mapeq(a, dumpload(a), Teq))
end
function test.nested_deepcopy()
    local common = paths.dofile('../../utils/common.lua')
    local function r()
        return torch.random()
    end
    local function rt()
        local size = common.rand_int_table(1, 4, 1, 4)
        return torch.rand(unpack(size))
    end
    local function Teq(a, b) return totem.areTensorsEq(a, b, 1e-5) end
    local df = nested.deepcopy
    local function _test(a, c, d)
        b = df(a, nil)
        cc = df(a, c)
        dd = df(a, d)
        tester:assert(nested.mapeq(a, b, Teq))
        tester:assert(nested.mapeq(a, cc, Teq))
        tester:assert(nested.mapeq(a, dd, Teq))
        a = nested.zero_nested(a, {gpuid=-1})
        tester:assert(not nested.mapeq(a, b, Teq))
        tester:assert(not nested.mapeq(a, c, Teq))
        tester:assert(not nested.mapeq(a, d, Teq))
        tester:assert(nested.mapeq(b, c, Teq))
        tester:assert(c == cc)
        tester:assert(d == dd)
    end
    a = Om{{[r()] = rt()}, {[r()] = rt()}}
    c = Om{{[a:keys()[1]] = rt()}, {[a:keys()[2]] = rt()}}
    d = Om{{[a:keys()[1]] = rt()}}
    _test(a, c, d)
    
    a = List{rt(), rt(), rt()}
    c = List{rt()}
    d = Om{{[2]=rt()}}
    _test(a, c, d)
    
    a = Om{{b=rt()}, {c=List{rt(), rt()}}, {a=rt()}}
    c = Om()
    d = Om({c=List()})
    _test(a, c, d)
end
function test.nested_nestmap2()
    local function r()
        return torch.random()
    end
    local function rt()
        return torch.rand(3, 4)
    end
    a = Om{{u = rt()}, {v = rt()}}
    aa = nested.deepcopy(a)
    c = Om{{u = rt()}, {v = rt()}}
    d = nested.nestadd(a, c)
    tester:assert(a == d)
    tester:assert(a.u == d.u and a.v == d.v)
    tester:assertTensorEq(aa.u + c.u, d.u, 1e-9)
    tester:assertTensorEq(aa.v + c.v, d.v, 1e-9)
    
    a = List{rt(), rt(), rt()}
    aa = nested.deepcopy(a)
    c = List{rt()}
    d = nested.nestadd(a, c)
    tester:assert(a == d)
    tester:assert(a[1] == d[1] and a[2] == d[2] and a[3] == d[3])
    tester:assertTensorEq(aa[1] + c[1], d[1], 1e-9)
    tester:assertTensorEq(aa[2], d[2])
    tester:assertTensorEq(aa[3], d[3])
    
    a = Om{{b=rt()}, {c=List{rt(), rt()}}, {a=rt()}}
    aa = nested.deepcopy(a)
    c = Om{{b=rt()}, {c=List{rt()}}, {a=rt()}, {e=rt()}}
    d = nested.nestadd(a, c)
    tester:assert(a == d)
    tester:assert(a.b == d.b)
    tester:assertTensorEq(aa.b + c.b, d.b, 1e-9)
    tester:assertTensorEq(aa.a + c.a, d.a, 1e-9)
    tester:assertTensorEq(aa.c[1] + c.c[1], d.c[1], 1e-9)
    tester:assertTensorEq(aa.c[2], d.c[2], 1e-9)
    tester:assert(d.e == nil)
end
    
tester:add(test):run()