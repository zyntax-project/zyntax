-- Closures: shared upvalues, per-iteration loop variables, recursion.
local function counter()
  local n = 0
  return function()
    n = n + 1
    return n
  end
end
local c1, c2 = counter(), counter()
print(c1(), c1(), c1(), c2())

local fns = {}
for i = 1, 3 do
  fns[i] = function() return i * 10 end
end
print(fns[1](), fns[2](), fns[3]())

local function adder(x)
  return function(y) return x + y end
end
local add5 = adder(5)
print(add5(10), adder(1)(2))

local shared = 0
local function inc() shared = shared + 1 end
local function get() return shared end
inc() inc() inc()
print(get(), shared)

local function fact(n)
  if n <= 1 then return 1 end
  return n * fact(n - 1)
end
print(fact(10))

local function outer()
  local a = 1
  local function middle()
    local b = 2
    local function inner()
      a = a + 1
      b = b + 1
      return a + b
    end
    return inner
  end
  return middle()
end
local f = outer()
print(f(), f())

local acc = {}
for _, name in ipairs({ "x", "y" }) do
  acc[#acc + 1] = function() return name end
end
print(acc[1](), acc[2]())

local function memoize(fn)
  local cache = {}
  return function(n)
    local v = cache[n]
    if v == nil then
      v = fn(n)
      cache[n] = v
    end
    return v
  end
end
local slow_calls = 0
local sq = memoize(function(n) slow_calls = slow_calls + 1 return n * n end)
print(sq(4), sq(4), sq(5), slow_calls)

local x = 10
local function readx() return x end
x = 20
print(readx())
local function selfref(n)
  local function helper(m)
    if m == 0 then return "done" end
    return helper(m - 1)
  end
  return helper(n)
end
print(selfref(3))
