-- debug.setlocal reaches a frame stopped at a call in any position.
local function setter(v) return debug.setlocal(2, 1, v) end

local function positions()
  local a = 1
  local r = setter(2)
  print("local", a, r)
  if setter(3) then print("condition", a) end
  print("argument", setter(4), a)
  local t = {setter(5)}
  print("constructor", a, #t)
  a = setter(6) and a
  print("assignment", a)
  while setter(7) do break end
  print("while", a)
  setter(8)
  print("statement", a)
  return a
end
print("returned", positions())

-- at level 1, from the frame itself
local function own()
  local b = 1
  print(debug.setlocal(1, 1, "str"))
  print(b)
  local r = debug.setlocal(1, 1, 5)
  print(b, r)
end
own()

-- through a closure called in an initializer
local function outer()
  local z = 1
  local w = (function() debug.setlocal(2, 1, 70) return 1 end)()
  print(z, w)
end
outer()

-- a numeric for's variable is a copy of the counter: setting it
-- changes the rest of the iteration, not the loop
for i = 1, 3 do
  local _, slot = nil, nil
  for k = 1, 10 do
    local name = debug.getlocal(1, k)
    if name == "i" then slot = k break end
  end
  debug.setlocal(1, slot, "x" .. i)
  print(i, math.type(i))
end
for i = 0, 0.25, 0.125 do
  io.write(tostring(i), " ")
end
print()
