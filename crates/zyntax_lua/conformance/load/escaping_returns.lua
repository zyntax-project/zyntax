-- What a loaded chunk returns is used outside it, with whatever its
-- callers pass.
local eq = load("return function(x, y) return x == y end")()
print(eq(3, 100), eq(3, 3), eq(5.0, 5), eq("a", "a"), eq({}, {}))
local ops = load("return function(x, y) return x ~= y, x < y, x .. '', y end")()
print(ops(1, 2))
print(ops("a", "b"))
local mk = load("local k = ... return { get = function() return k end }")
print(mk(42).get(), mk("s").get())
local tab, id = load("return {f = function(a) return a * 2 end}, function(b) return b end")()
print(tab.f(21), tab.f(1.5), id(nil), id("x"))
