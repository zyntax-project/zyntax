-- Recursion, direct calls, multiple returns, varargs.
function fib(n)
  if n < 2 then return n end
  return fib(n - 1) + fib(n - 2)
end
print(fib(20))

local function divmod(a, b)
  return a // b, a % b
end
local q, r = divmod(17, 5)
print(q, r)
print(divmod(9, 4))
print((divmod(9, 4)))

local function sum(...)
  local total = 0
  for i = 1, select("#", ...) do
    total = total + select(i, ...)
  end
  return total
end
print(sum(1, 2, 3, 4), sum(), sum(10))

local function pack2(...)
  local t = { ... }
  return #t, t[1], t[#t]
end
print(pack2("a", "b", "c"))

function noret() end
print(noret())
print((noret()))
print(type((noret())))

local function three() return 1, 2, 3 end
print(three(), "x")
print("x", three())
local a, b, c, d = three()
print(a, b, c, d)
local t = { three(), three() }
print(#t)

function early(x)
  if x > 0 then return "pos" end
  if x < 0 then return "neg" end
end
print(early(1), early(-1), early(0))

local function apply(f, x) return f(x) end
print(apply(function(v) return v * 2 end, 21))
print(apply(fib, 10))
local g = fib
print(g(15))
print(select(2, "a", "b", "c"))
print(select(-1, "a", "b", "c"))
