-- load(string.dump(f)) gives a function with fresh upvalues: the
-- first holds the environment, the rest are nil.

local function add(a, b) return a + b end
local f = load(string.dump(add))
print(type(f), f(2, 3))

-- varargs and several results
local function pack(...) return select("#", ...), ... end
local g = load(string.dump(pack))
print(g(1, nil, 3))

-- nested functions keep working after the round trip
local function curry(x)
  return function (y)
    return function (z) return x + y + z end
  end
end
local c = load(string.dump(curry))
print(c(1)(2)(3))

-- a method keeps its self
local obj = {n = 7}
function obj:get(k) return self.n + (k or 0) end
local m = load(string.dump(obj.get))
print(m(obj), m(obj, 3))

-- upvalues are fresh: the first is the environment, the rest nil
local a, b = 20, 30
local function uses() return a, b end
local u = load(string.dump(uses))
print(debug.getupvalue(u, 1) == "a", select(2, debug.getupvalue(u, 1)) == _ENV)
print(debug.getupvalue(u, 2))
print(debug.getupvalue(u, 3))
print(debug.setupvalue(u, 1, "one"), debug.setupvalue(u, 2, "two"))
print(u())
print(a, b)

-- the environment where it is not the first upvalue
local x = 5
XX = 123
local function h()
  local y = x
  return XX
end
local hd = load(string.dump(h), "", "b")
print(debug.getupvalue(hd, 1), debug.getupvalue(hd, 2))
print(pcall(hd))
debug.setupvalue(hd, 2, _G)
print(hd())
XX = nil

-- assigned upvalues keep being shared between the loaded function's calls
local p, q = 1, 2
local w = load(string.dump(function (v)
  if v ~= "set" then return p end
  p = 10 + q; q = q + 1
end))
print(w() == _ENV)
debug.setupvalue(w, 1, 0)
debug.setupvalue(w, 2, 13)
w("set")
print(w())
w("set")
print(w())

-- a main chunk, dumped and loaded again, runs with the environment
local main = load("local t = ... ; return (t or 0) + 1, _ENV == _G")
local md = load(string.dump(main))
print(md(41))
print(debug.getupvalue(md, 1) == "_ENV")
local env = {print = print}
local me = load(string.dump(load("seen = 1; return seen")), nil, "b", env)
print(me(), env.seen, seen)

-- a dump of a loaded dump
local again = load(string.dump(load(string.dump(add))))
print(again(4, 5))

-- strings with any byte survive
local s = load(string.dump(function () return "\0alo\255", "\r\n" end))
local s1, s2 = s()
print(#s1, s1:byte(1, -1))
print(#s2)

-- line numbers are the original's
local function fails()
  local t = nil
  return t.x
end
print(select(2, pcall(load(string.dump(fails)))))
