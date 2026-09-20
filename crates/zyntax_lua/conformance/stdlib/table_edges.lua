-- table.move, unpack and sort at the edges
local maxI, minI = math.maxinteger, math.mininteger
local function eqT(a, b)
  for k, v in pairs(a) do assert(b[k] == v) end
  for k, v in pairs(b) do assert(a[k] == v) end
  return true
end
print(pcall(table.move, 1, 2, 3, 4))
print(eqT(table.move({10, 20, 30}, 1, 3, 2), {10, 10, 20, 30}))
print(eqT(table.move({10, 20, 30}, 1, 3, 3), {10, 20, 10, 20, 30}))
local a = {10, 20, 30, 40}
table.move(a, 1, 4, 2, a)
print(eqT(a, {10, 10, 20, 30, 40}))
print(eqT(table.move({10, 20, 30}, 2, 3, 1), {20, 30, 30}))
a = {}
print(table.move({10, 20, 30}, 1, 3, 1, a) == a, eqT(a, {10, 20, 30}))
a = {}
print(table.move({10, 20, 30}, 1, 0, 3, a) == a, eqT(a, {}))
print(eqT(table.move({10, 20, 30}, 1, 10, 1), {10, 20, 30}))
a = table.move({[maxI - 2] = 1, [maxI - 1] = 2, [maxI] = 3}, maxI - 2, maxI, -10, {})
print(eqT(a, {[-10] = 1, [-9] = 2, [-8] = 3}))
a = table.move({[minI] = 1, [minI + 1] = 2, [minI + 2] = 3}, minI, minI + 2, -10, {})
print(eqT(a, {[-10] = 1, [-9] = 2, [-8] = 3}))
a = table.move({45}, 1, 1, maxI)
print(eqT(a, {45, [maxI] = 45}))
a = table.move({[maxI] = 100}, maxI, maxI, minI)
print(eqT(a, {[minI] = 100, [maxI] = 100}))
-- through metamethods
a = setmetatable({}, {__index = function (_, k) return k * 10 end, __newindex = error})
local b = table.move(a, 1, 10, 3, {})
print(eqT(a, {}), eqT(b, {nil, nil, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}))
b = setmetatable({""}, {
  __index = error,
  __newindex = function (t, k, v) t[1] = string.format("%s(%d,%d)", t[1], k, v) end})
table.move(a, 10, 13, 3, b)
print(b[1])
local stat, msg = pcall(table.move, b, 10, 13, 3, b)
print(stat, msg == b)
-- a very long move is stopped by the first error
local function checkmove(f, e, t, x, y)
  local pos1, pos2
  local a = setmetatable({}, {
    __index = function (_, k) pos1 = k end,
    __newindex = function (_, k) pos2 = k; error() end})
  local st, msg = pcall(table.move, a, f, e, t)
  print(st, msg, pos1 == x, pos2 == y)
end
checkmove(1, maxI, 0, 1, 0)
checkmove(0, maxI - 1, 1, maxI - 1, maxI)
checkmove(minI, -2, -5, -2, maxI - 6)
checkmove(minI + 1, -1, -2, -1, maxI - 3)
checkmove(minI, -2, 0, minI, 0)
checkmove(minI + 1, -1, 1, minI + 1, 1)
print(select(2, pcall(function () return table.move({}, 0, maxI, 1) end)))
print(select(2, pcall(function () return table.move({}, -1, maxI - 1, 1) end)))
print(select(2, pcall(function () return table.move({}, minI, -1, 1) end)))
print(select(2, pcall(function () return table.move({}, minI, maxI, 1) end)))
print(select(2, pcall(function () return table.move({}, 1, maxI, 2) end)))
print(select(2, pcall(function () return table.move({}, 1, 2, maxI) end)))
print(select(2, pcall(function () return table.move({}, minI, -2, 2) end)))
-- unpack at the ends of the integers
local unpack = table.unpack
print(unpack({[maxI] = 20}, maxI, maxI))
print(unpack({[maxI - 1] = 12, [maxI] = 23}, maxI - 1, maxI))
print(unpack({[minI] = 12.3, [minI + 1] = 23.5}, minI, minI + 1))
print(unpack({[minI] = 12.3}, minI, minI))
print(select("#", unpack({}, minI + 1, minI)))
print(pcall(unpack, {}, 0, maxI))
print(pcall(unpack, {}, 1, maxI))
print(select("#", unpack({}, maxI, 0)))
-- sort: an order that is not one, and lengths that are not
local function f(x, y) assert(x and y); return true end
print(pcall(table.sort, {1, 2, 3, 4}, f))
print(pcall(table.sort, {1, 2, 3, 4, 5, 6}, f))
a = setmetatable({}, {__len = function () return -1 end})
print(#a)
table.sort(a, error)
a = setmetatable({}, {__len = function () return maxI end})
print(pcall(function () return table.sort(a) end))
local t = {5, 3, 8, 1, 9, 2, 7}
table.sort(t)
print(table.concat(t, ","))
table.sort(t, function (x, y) return x > y end)
print(table.concat(t, ","))
