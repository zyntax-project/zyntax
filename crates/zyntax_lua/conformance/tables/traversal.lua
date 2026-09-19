-- fields may be cleared during a traversal
local function clear(t)
  local n = 0
  for k in pairs(t) do t[k] = nil; n = n + 1 end
  return n, next(t)
end
print(clear({1, 2, 3}))
print(clear({a = 1, b = 2, c = 3, d = 4}))
print(clear({1, 2, 3, a = 1, b = 2}))
print(clear({1, 2, 3, 4, 5, x = 1, y = 2, z = 3}))
local t = {1, 2, 3, x = 1}
local seen = {}
for k, v in next, t do
  t[k] = nil
  seen[#seen + 1] = tostring(k)
end
table.sort(seen)
print(table.concat(seen, " "), next(t))
t = {1, 2, 3}
t[3] = nil
print(next(t, 2))
print(pcall(next, {}, "nokey"))
print(pcall(next, {a = 1}, "b"))
t = {1, 2, 3, x = 1}
t[2] = nil
print(next(t, 1), next(t, 3))
