-- next with a key the table never had raises; one removed during a
-- traversal does not
print(pcall(next, {10, 20}, 3))
print(pcall(next, {10, 20}, 0))
print(pcall(next, {10, 20}, "x"))
print(pcall(next, {}, 1))
print(next({10, 20}, 2), next({10, 20}, 1))
print(next({10, 20, x = 1}, 2))
-- clearing while traversing, from the front and from the back
local t = {1, 2, 3, 4, 5, a = 1, b = 2}
local n = 0
for k in pairs(t) do t[k] = nil; n = n + 1 end
print(n, next(t), #t)
t = {1, 2, 3, 4, 5}
local k = next(t)
while k do
  local nk = next(t, k)
  t[k] = nil
  k = nk
end
print(next(t), #t)
-- the last element removed, then next from its key
t = {1, 2, 3}
t[3] = nil
print(next(t, 3), next(t, 2), #t)
print(pcall(next, t, 4))
-- table.remove, then next from the old last key
t = {1, 2, 3}
table.remove(t)
print(next(t, 3), #t)
print(pcall(next, t, 4))
-- a constructor's trailing nil was a slot
t = {1, nil}
print(next(t, 2))
print(pcall(next, t, 3))
-- grown again after the removal
t = {1, 2, 3}
t[3] = nil
t[3] = 30
t[4] = 40
print(next(t, 4), #t)
print(pcall(next, t, 5))
