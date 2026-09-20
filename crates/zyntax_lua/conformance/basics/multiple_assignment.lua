-- every target's table and key, and every value, is evaluated before a store
local a, i, j, b
a = {'a', 'b'}; i = 1; j = 2; b = a
i, a[i], a, j, a[j], a[i + j] = j, i, i, b, j, i
print(i, b[1], a, j == b, b[2], b[3])
a = {}
local function foo()    -- assigning to upvalues
  b, a.x, a = a, 10, 20
end
foo()
print(a, b.x, b == b)
local t = {1, 2}
t[1], t[2] = t[2], t[1]
print(t[1], t[2])
local x, y = 1, 2
x, y = y, x
print(x, y)
local n = 0
local function next_n() n = n + 1; return n end
local u = {}
u[next_n()], u[next_n()] = next_n(), next_n()
print(u[1], u[2], n)
