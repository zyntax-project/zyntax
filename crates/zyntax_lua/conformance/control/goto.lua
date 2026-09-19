-- continue idiom in every loop kind
local out = {}
for i = 1, 6 do
  if i % 2 == 0 then goto continue end
  out[#out + 1] = i
  ::continue::
end
print(table.concat(out, ","))
local t = { a = 1, b = 2, c = 3 }
local n = 0
for k, v in pairs(t) do
  if v == 2 then goto skip end
  n = n + v
  ::skip::
end
print(n)
local i = 0
while i < 5 do
  i = i + 1
  if i == 3 then goto next end
  io.write(i, " ")
  ::next::
end
print()
i = 0
repeat
  i = i + 1
  if i == 2 then goto again end
  io.write(i, " ")
  ::again::
until i >= 4
print()

-- forward jump over statements
do
  goto done
  print("never")
  ::done::
  print("done")
end

-- backward jump: a loop from gotos
local k = 0
::top::
k = k + 1
if k < 3 then goto top end
print("k", k)

-- goto out of nested loops
for a = 1, 3 do
  for b = 1, 3 do
    if a * b == 4 then goto found end
  end
end
print("not found")
::found::
print("found")

-- goto to an enclosing block's label from inside an if
local function f(x)
  if x > 0 then
    if x > 10 then goto big end
    return "small"
  end
  do return "nonpositive" end
  ::big::
  return "big"
end
print(f(1), f(11), f(-1))

-- state machine
local function sm(s)
  local out = {}
  local pos = 1
  ::start::
  local c = s:sub(pos, pos)
  if c == "" then goto finish end
  if c == "a" then out[#out + 1] = "A"; pos = pos + 1; goto start end
  out[#out + 1] = c; pos = pos + 1
  goto start
  ::finish::
  return table.concat(out)
end
print(sm("abcab"))

-- same label name in sibling blocks
do goto x; ::x:: io.write("1 ") end
do goto x; ::x:: io.write("2 ") end
print()

-- closure in a goto loop captures fresh locals
local fs = {}
local j = 1
::loop::
do
  local v = j * 10
  fs[j] = function() return v end
end
j = j + 1
if j <= 3 then goto loop end
print(fs[1](), fs[2](), fs[3]())
