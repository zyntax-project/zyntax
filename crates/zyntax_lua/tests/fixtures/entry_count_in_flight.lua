local function f(n, t)
  local s = 0
  for i = 1, n do
    s = s + (i ~ t.k) % 7
    if i % 5000011 == 0 then t.k = t.k + 1 end
  end
  local s2 = 0
  for i = 1, n do s2 = s2 + i % 3 end
  return s * 3 + s2
end
local t = {k = 1}
local acc = 0
for r = 1, 3000 do acc = acc + f(10, t) end
acc = acc + f(60000000, t)
print(acc, t.k)
