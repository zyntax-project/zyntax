local f = load("return 1 + 2")
print(f())
print(load("x = 10; return x")())
print(x)
local add = load("local a, b = ...; return a + b")
print(add(3, 4))
counter = 0
load("counter = counter + 1")()
load("counter = counter + 1")()
print(counter)
local env = { y = 5 }
local h = load("y = y * 2; return y", "chunk", "t", env)
print(h(), env.y, y)
print(type(load("error('inside')", "=named")))
print(pcall(load("error('inside')", "=named")))
print(pcall(load("error('inside')")))
local fact = load([[
  local function fact(n)
    if n <= 1 then return 1 end
    return n * fact(n - 1)
  end
  return fact
]])()
print(fact(10))
print(load("return ...", "vararg")(1, 2, 3))
local s = "return function(a) return a * 3 end"
print(load(s)()(7))
print(type(load), load == load)
print(load("return string.rep('ab', 2)")())
