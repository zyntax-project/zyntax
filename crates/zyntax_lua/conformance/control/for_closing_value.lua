-- A generic for's fourth value is closed however the loop is left.
local function closer(name, log)
  return setmetatable({}, {__close = function (_, err)
    log[#log + 1] = name .. ":" .. tostring(err)
  end})
end

-- Normal end.
do
  local flag = false
  local x = setmetatable({}, {__close = function () flag = true end})
  local function a () return (function () return nil end), nil, nil, x end
  for k in a() do end
  print("normal end", flag)
end

local function range(n, c)
  local i = 0
  return function () i = i + 1; if i <= n then return i end end, nil, nil, c
end

-- break, return, error.
local log = {}
for i in range(3, closer("a", log)) do
  if i == 2 then break end
end
local function ret ()
  for i in range(3, closer("b", log)) do
    if i == 2 then return i end
  end
end
ret()
print(pcall(function ()
  for i in range(3, closer("c", log)) do
    if i == 2 then error("boom", 0) end
  end
end))
for i in range(2, closer("d", log)) do end
print(table.concat(log, " "))

-- goto out of nested loops closes both; a goto within the body does not.
local numopen = 0
local function counted ()
  numopen = numopen + 1
  return setmetatable({}, {__close = function () numopen = numopen - 1 end})
end
for i in range(3, counted()) do
  for j in range(3, counted()) do
    if j == 2 then goto continue end
    if i == 2 and j == 3 then goto out end
    ::continue::
  end
end
::out::
print("numopen", numopen)

-- The closing value is closed after the body's own variables.
local order = {}
for i in range(1, closer("state", order)) do
  local v <close> = closer("body", order)
  break
end
print(table.concat(order, " "))

-- A return in the loop is not a tail call: the value closes after it.
local closed = false
local function foo ()
  return function () return true end, 0, 0,
         setmetatable({}, {__close = function () closed = true end})
end
local function tail () return closed end
local function foo1 ()
  for k in foo() do return tail() end
end
print("tail sees", foo1(), closed)

-- next, t, nil, closing value.
local o1 = setmetatable({}, {__close = function () print("o1 closed") end})
for k, v in next, {}, nil, o1 do
  local function f () return k end
  break
end

-- nil and false are not closed; three values close nothing.
for k in next, {}, nil, nil do end
for k in next, {}, nil, false do end
for k, v in pairs({1}) do end
print("done")
