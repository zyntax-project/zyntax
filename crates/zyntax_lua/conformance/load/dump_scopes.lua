-- Dumped functions keep what their text alone does not say: folded
-- constants, a local named _ENV, labels and recursion.

local N <const> = 10
local M <const> = N
local T <const> = "t\0"
local S <const> = "s" .. "t"
local function consts() return N, M, #T, S end
local c = load(string.dump(consts))
local n, m, t, s = c()
print(n, m, t, s == _ENV)
print(debug.getupvalue(c, 1) == "S")

-- a constant shadowed by a later local is that local, an upvalue
local K <const> = 3
local function uses_k() return K end
local K = 4
local function uses_k2() return K, N end
print(load(string.dump(uses_k))(), (debug.getupvalue(load(string.dump(uses_k2)), 1)))
local k2 = load(string.dump(uses_k2))
debug.setupvalue(k2, 1, 7)
print(k2())

-- a local named _ENV captured by the function
do
  local dump, load, print, getupvalue, setupvalue =
    string.dump, load, print, debug.getupvalue, debug.setupvalue
  local G = _G
  local _ENV = {x = "inner"}
  local function reads() return x end
  local r = load(dump(reads))
  local name, value = getupvalue(r, 1)
  print(name, getupvalue(r, 2), value == G)
  setupvalue(r, 1, {x = "set"})
  print(r())
end

-- _ENV as the second upvalue, then written through
local y = 1
local function writes()
  local v = y
  G_written = v
end
local w = load(string.dump(writes))
print(debug.getupvalue(w, 1) == "y", debug.getupvalue(w, 2))
debug.setupvalue(w, 1, 5)
debug.setupvalue(w, 2, _G)
w()
print(G_written)
G_written = nil

-- goto and labels inside the dumped function
local function loops(n)
  local i, s = 0, 0
  ::top::
  i = i + 1
  if i > n then goto done end
  s = s + i
  goto top
  ::done::
  return s
end
print(load(string.dump(loops))(10))

-- a recursive local function: its own upvalue is fresh
local function fact(n) if n <= 1 then return 1 end return n * fact(n - 1) end
local lf = load(string.dump(fact))
print(debug.getupvalue(lf, 1) == "fact", lf(1))
debug.setupvalue(lf, 1, fact)
print(lf(5))

-- a function spanning lines keeps its line numbers in errors
local function late(t)


  return t.a.b
end
print(pcall(load(string.dump(late)), {}))
