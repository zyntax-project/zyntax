-- Constructor entries under keys computed at run time: read back
-- through `or`, `== nil` and a walk.
local k = ("a"):rep(1)
local t = {[k] = 5}
print((t[k] or 0) + 1, t[k] == nil, t[k] ~= nil, type(t[k]))
local x = t[k]
if x == nil then print("absent") else print("present", x) end

local b = {[true] = 5}
print((b[true] or 0) + 1, b[false] == nil)

local obj = {}
local v = {[obj] = 3}
print((v[obj] or 0) + 1, v[{}] == nil)

local one = 1
local n = {[one] = 5}
print((n[1] or 0) + 1, (n[one] or 0) + 1, n[one] == nil)
local u = {[2] = 7}
local i = 2
print((u[i] or 0) + 1, u[i + 1] == nil)

local ks = {"p", "q"}
local w = {[ks[1]] = 4, [ks[2]] = 5}
local sum = 0
for _, kk in ipairs(ks) do sum = sum + (w[kk] or 0) end
local cnt = 0
for _, val in pairs(w) do cnt = cnt + val end
print(sum, cnt)

-- A computed key naming a field of the same constructor.
local f = ("x"):rep(1)
local r = {x = 1, [f] = 2.5}
print(r.x + 1)
local s = {x = 1, y = 2, [f] = "s"}
print(r.x, s.x, s.y)

-- Mixed with positional values.
local m = {10, 20, [k] = 3, [one + 2] = 30}
local total = 0
for j = 1, 3 do total = total + (m[j] or 0) end
print(total, (m[k] or 0) * 2)

-- A table born with a metatable.
local mt = {__index = function(_, key) return 100 end}
local h = setmetatable({[k] = 5}, mt)
print(h[k] + 1, h.zz + 1)
