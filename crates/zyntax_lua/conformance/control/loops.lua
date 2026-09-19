-- Control flow: if chains, while, repeat, numeric for, break, goto continue.
local function classify(n)
  if n < 0 then
    return "negative"
  elseif n == 0 then
    return "zero"
  elseif n < 10 then
    return "small"
  else
    return "large"
  end
end
print(classify(-5), classify(0), classify(7), classify(100))

local i = 0
while i < 5 do
  i = i + 1
  if i == 3 then break end
end
print(i)

local n = 0
repeat
  n = n + 1
  local done = n >= 4
until done
print(n)

for i = 1, 5 do io.write(i, " ") end
print()
for i = 10, 1, -3 do io.write(i, " ") end
print()
for i = 1, 0 do print("never") end
for x = 0.5, 2.5, 0.5 do io.write(x, " ") end
print()
for i = 1, 3 do
  for j = 1, 3 do
    if j == 2 then goto continue end
    io.write(i, ":", j, " ")
    ::continue::
  end
end
print()

local total = 0
for i = 1, 100 do
  if i % 2 == 0 then goto skip end
  total = total + i
  ::skip::
end
print(total)

local k = 1
while true do
  k = k * 2
  if k > 1000 then break end
end
print(k)

local v = nil
if v then print("truthy") else print("falsy") end
if 0 then print("zero is true") end
if "" then print("empty string is true") end
if not nil then print("not nil") end
print(nil and 1, false or "default", 1 and 2, nil or false, false and nil)
print(1 and nil, "a" or error("never"))
local x = 5
local y = x > 3 and "big" or "small"
print(y)
do
  local x = 10
  print(x)
end
print(x)
for i = 3, 1 do print("no") end
local count = 0
for i = math.maxinteger - 2, math.maxinteger do count = count + 1 end
print(count)
