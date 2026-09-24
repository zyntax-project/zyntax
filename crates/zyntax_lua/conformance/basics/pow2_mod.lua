-- Integer floor division and modulo by powers of two, for either sign,
-- against divisors that are not.

local function row(label, first, last, step)
  local out = {}
  for i = first, last, step or 1 do
    out[#out + 1] = (i % 8) .. ":" .. (i // 4)
  end
  print(label, table.concat(out, " "))
end
row("ascending", 0, 12)
row("descending", 12, -12, -1)
row("negative start", -17, 3)
row("step 2", -9, 9, 2)
row("step 3", -20, 20, 3)

local t = { 10, 20, 30, 40, 50, 60, 70, 80, 90 }
local out = {}
for i = 1, #t do
  out[#out + 1] = t[i] % 16 + i % 2 + i // 2
end
print("length bound", table.concat(out, " "))

local sum = 0
for i = 0, 999 do
  sum = sum + i % 8
end
print("sum", sum)

-- Divisors 1, 2 and a large power.
for _, x in ipairs({ -7, -1, 0, 1, 7 }) do
  print(x, x % 1, x // 1, x % 2, x // 2, x % 1024, x // 1024, x % 4611686018427387904,
    x // 4611686018427387904)
end
print(math.mininteger % 8, math.mininteger // 8, math.maxinteger % 16, math.maxinteger // 16)
print(math.mininteger // 1, math.mininteger % 2, math.maxinteger // 4611686018427387904)

-- Floats keep float results.
for i = -2.5, 2.5 do
  io.write(i % 8, " ", i // 4, "  ")
end
print()
for x = 0.5, 20.5, 6.5 do
  io.write(x % 4, " ", x // 2, "  ")
end
print()

-- A number that is an integer or a float, known only at run time.
local v = 7
for i = 1, 3 do
  print(v % 4, v // 4, -v % 4, -v // 4)
  v = v + 0.5
end

-- Divisors that are not powers of two, or not literals.
local d = 8
for _, x in ipairs({ -9, -8, -1, 0, 1, 8, 9 }) do
  io.write(x % 6, " ", x // 6, " ", x % -8, " ", x // -8, " ", x % d, " ", x // d, "  ")
end
print()
print(pcall(function() return 1 % 0 end))
print(pcall(function() return 1 // 0 end))
