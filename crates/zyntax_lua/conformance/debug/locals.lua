-- debug.getlocal and setlocal over live frames.
local function f(a, b)
  local c = a + b
  local names = {}
  local i = 1
  while true do
    local name, value = debug.getlocal(1, i)
    if not name then break end
    if type(value) == "table" then value = "{}" end
    names[#names + 1] = name .. "=" .. tostring(value)
    i = i + 1
  end
  return table.concat(names, " ")
end
print(f(1, 2))

local function g(x)
  local y = x * 2
  local function inner()
    print(debug.getlocal(2, 1))
    print(debug.getlocal(2, 2))
    print(debug.setlocal(2, 2, 100))
  end
  inner()
  return y
end
print(g(5))

print(debug.getlocal(f, 1), debug.getlocal(f, 2), debug.getlocal(f, 3))

local function va(...)
  return debug.getlocal(1, -1), debug.getlocal(1, -2)
end
print(va("first", "second"))

local function loop()
  local before = 0
  for k = 10, 10 do
    print(debug.getlocal(1, 1))
    print((debug.getlocal(1, 2)))
    print(debug.getlocal(1, 5))
  end
  for key, value in pairs({x = 1}) do
    print(debug.getlocal(1, 6), debug.getlocal(1, 7))
  end
end
loop()

local ok, err = pcall(debug.getlocal, 50, 1)
print(ok, err:match("%((.-)%)$"))
