local function pick(k)
  if k % 2 == 0 then return k end
  return k + 0.5
end
for k = -3, 6 do
  local n = pick(k)
  local f = 1.5
  print(k, math.type(n), n + f, n - f, f - n, n * f, n // f, f // n, n % f, f % n, n // 0.5, n % 0.5, n % 2.0, -n * 0.0)
  print(math.type(n + 1.0), math.type(n * 2.0), math.type(n // 2.0), math.type(n % 2.0))
  print(n + f > 3, n - f == 0, (n + f) // 1 | 0 == 1)
end
local z = pick(0)
print(z // 0.0, z % 0.0, -z // 0.0, 0.0 // z, z * (1/0), z + (0/0))
print(pick(2) % -1.5, pick(3) % -1.5, pick(2) // -1.5, 5.0 % pick(3))
print(math.maxinteger + 0.0 == pick(4) * 2.0, pick(math.maxinteger) + 1.0)
