-- upvaluejoin on a function loaded from a dump with many upvalues.

local nup = 200
local prog = {"local a1"}
for i = 2, nup do prog[#prog + 1] = ", a" .. i end
prog[#prog + 1] = " = 1"
for i = 2, nup do prog[#prog + 1] = ", " .. i end
local sum = 1
prog[#prog + 1] = "; return function () return a1"
for i = 2, nup do prog[#prog + 1] = " + a" .. i; sum = sum + i end
prog[#prog + 1] = " end"
prog = table.concat(prog)
local f = assert(load(prog))()
print(f() == sum)

f = load(string.dump(f))
print(debug.getupvalue(f, 1), select(2, debug.getupvalue(f, 1)) == _ENV)
print(debug.getupvalue(f, nup))
print(debug.getupvalue(f, nup + 1))
local a = 10
local h = function () return a end
for i = 1, nup do
  debug.upvaluejoin(f, i, h, 1)
end
print(f() == 10 * nup, f())
a = 3
print(f())

-- joined upvalues of a small loaded function share the cell
local x, y = 1, 2
local g = load(string.dump(function () x = x + 1; return x + y end))
local k = 100
local get = function () return k end
debug.upvaluejoin(g, 1, get, 1)
debug.upvaluejoin(g, 2, get, 1)
print(g(), k)
print(debug.upvalueid(g, 1) == debug.upvalueid(get, 1))
