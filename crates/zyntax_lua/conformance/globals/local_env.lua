-- a local named _ENV scopes the free names inside it
x = "global x"
do
  local _ENV = {print = print, x = "env x"}
  print(x)
  x = "written"
  print(x)
  y = 5
  print(y, _ENV.y)
end
print(x, y)
-- functions defined inside capture the environment as an upvalue
local function make()
  local _ENV = {assert = assert, tostring = tostring, n = 0}
  local function bump() n = n + 1; return n end
  function stored() return "stored in the env" end
  return bump, _ENV
end
local bump, env = make()
print(bump(), bump(), env.n, env.stored(), stored)
-- a non-table environment: the free names index it
local print, pcall, select = print, pcall, select
do
  local _ENV = 11
  print(pcall(function () X = 1 end))
  print(pcall(function () return X end))
end
print(select(2, pcall(function () local _ENV = nil; return z end)))
print(select(2, pcall(function () local _ENV = nil; z = 1 end)))
-- shadowed builtins are not reachable, an empty environment sees nothing
do
  local _ENV = {}
  print(pcall(function () return print == nil, string end))
end
local ok, err = pcall(function () local _ENV = {}; print("x") end)
print(ok, err)
-- _G and _ENV inside are fields as well
do
  local _ENV = {print = print, _G = "not the globals"}
  print(_G, _ENV._G)
  _G = "changed"
  print(_G)
end
print(type(_G))
-- nested: the innermost _ENV wins
do
  local _ENV = {print = print, v = "outer"}
  print(v)
  do
    local _ENV = {print = print, v = "inner"}
    print(v)
  end
  print(v)
end
-- an environment that is a proxy
do
  local seen = {}
  local _ENV = setmetatable({print = print, seen = seen}, {
    __index = function (_, k) seen[#seen + 1] = k; return k .. "!" end,
    __newindex = function (_, k, v) rawset(seen, k, v) end,
  })
  print(a, b)
  c = 3
  print(seen[1], seen[2], seen.c)
end
