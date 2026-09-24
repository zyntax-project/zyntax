-- a numeric for whose start or step is a float known only at run time is a float loop
local function half(w) return w / 2 end
for x = half(7), 5 do io.write(x, " ") end
print()
local x = 7
if #arg >= 0 then x = x / 2 end
for i = x, 5 do io.write(i, " ") end
print()
local step = 1
if #arg >= 0 then step = 0.5 end
for i = 1, 3, step do io.write(i, " ") end
print()
local s = 1
if #arg < 0 then s = 1.5 end
for i = s, 3 do io.write(math.type(i), " ") end
print()
local w = tonumber("5.0")
for i = w / 2, 4 do io.write(i, " ") end
print()
-- start and step from dynamic values: integers count in integers,
-- anything else in floats, a numeral string included
local mixed = {2, 2.5, "3"}
for k = 1, 3 do
  local out = {}
  for i = mixed[k], 4 do out[#out + 1] = tostring(i) end
  print(k, table.concat(out, " "))
end
for i = mixed[1], 1, -1 do io.write(i, " ") end
print()
for i = mixed[2], 0, -1 do io.write(i, " ") end
print()
for i = 1, 2, mixed[2] - 2 do io.write(i, " ") end
print()
print(pcall(function() for i = mixed[4], 2 do end end))
print(pcall(function() for i = mixed[1], 2, mixed[1] - 2 do end end))
print(pcall(function() for i = mixed[1], {}, 1 do end end))
-- an integer loop at the top of the range does not wrap
local count = 0
for i = math.maxinteger - 2, math.maxinteger, mixed[1] - 1 do count = count + 1 end
print(count)
count = 0
for i = mixed[1], 1e300 < 0 and 0 or math.huge, math.maxinteger do count = count + 1 end
print(count)
