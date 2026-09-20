-- functions as values: called through locals, upvalues, parameters,
-- returns, tables, the library, and after reassignment
local step = function(acc, i) return acc + i % 8 end
local function make_step(k) return function(acc, i) return acc + (i + k) % 8 end end
local step3 = make_step(3)
local function apply(f, v) return f(v) end
local function twice(f, v) return f(f(v)) end
local inc = function(x) return x + 1 end
local total = 0
for i = 0, 20 do total = step(total, i) + step3(0, i) end
print(total, apply(inc, 1), twice(inc, 1), apply(function(x) return x * 2 end, 21))
-- a function value that varies: two functions in one variable
local pick = inc
if total > 0 then pick = function(x) return x - 1 end end
print(pick(10), apply(pick, 10))
-- through a table, pcall, sort, and as an iterator
local t = { f = inc, g = step3 }
print(t.f(5), t.g(1, 2), pcall(inc, 41), select("#", pcall(step, 1, 2)))
local xs = { 5, 2, 8, 1 }
table.sort(xs, function(a, b) return a > b end)
print(table.concat(xs, ","))
local cmp = function(a, b) return a < b end
table.sort(xs, cmp)
print(table.concat(xs, ","))
local function range(n) local i = 0 return function() i = i + 1 if i <= n then return i end end end
local s = 0
for v in range(4) do s = s + v end
print(s)
-- nil, false and non-numbers reach a function stored in a table
local checks = { is = function(x) if x then return "truthy" else return "falsy" end end }
print(checks.is(nil), checks.is(false), checks.is(0), checks.is(""))
-- a closure whose captured variable changes
local counter = 0
local bump = function() counter = counter + 1 return counter end
print(bump(), bump(), apply(bump, nil), counter)
-- recursion through an upvalue, and mutual recursion through locals
local fact
fact = function(n) if n <= 1 then return 1 end return n * fact(n - 1) end
print(fact(10))
local even, odd
function even(n) if n == 0 then return true end return odd(n - 1) end
function odd(n) if n == 0 then return false end return even(n - 1) end
print(even(10), odd(7), apply(even, 3))
-- functions in conditions, comparisons and or/and
print(inc == inc, inc == step, inc ~= nil, (inc and 1) or 2, (nil or inc)(1), type(inc))
-- varargs into and out of a known function
local function pack2(...) return select("#", ...), ... end
print((pack2(inc, step)), select(2, pack2(inc, step)) == inc)
local function callit(f, ...) return f(...) end
print(callit(step, 1, 2), callit(inc, 9), callit(function(...) return select("#", ...) end, 1, 2, 3))
-- a function returned in several values and multiple assignment
local function pair() return inc, step end
local a, b = pair()
print(a(1), b(1, 2))
-- passed to coroutine.wrap and string.gsub
local co = coroutine.wrap(function(x) local y = coroutine.yield(x + 1) return y * 2 end)
print(co(1), co(10))
print(("abc"):gsub("%w", function(c) return c:upper() end))
-- a function value compared after being copied
local same = inc
print(same == inc, same(1) == inc(1), make_step(1) == make_step(1))
