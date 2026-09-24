local function busy(n)
  local x = 0
  for i = 1, n do x = x + (i ~ 5) % 3 end
  return x
end
local function f(n)
  local s = 0
  for i = 1, 3000 do s = s + i % 7 end
  s = s + busy(n * 4)
  local k, s2 = 0, 0
  local t = {v = 1}
  while k < n do
    s2 = s2 + (k ~ s) % 7 + t.v
    if k % 7000001 == 0 then t.v = t.v + 1 end
    k = k + 1
  end
  return s * 7 + s2
end
print(f(50000000))
