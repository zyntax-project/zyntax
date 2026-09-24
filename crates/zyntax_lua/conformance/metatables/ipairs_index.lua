-- ipairs reads through __index where the table has no element

local proto = { function(x) return x * 2 end }
print(proto[1](3))
local t = setmetatable({}, { __index = proto })
for i, v in ipairs(t) do
  print(i, v("4"))
  break
end

local calls = 0
local mt = {}
mt.__index = function(_, key)
  calls = calls + 1
  if key == "name" then return "handler" end
  if type(key) == "number" and key <= 2 then return key * 10 end
  return nil
end
local h = setmetatable({}, mt)
print(h.name)
for i, v in ipairs(h) do print(i, v) end
print(calls)

-- a class whose instances reach the runtime's own dispatch
local V = {}
V.__index = V
V.__add = function(a, b) return setmetatable({ x = a.x + b.x }, V) end
local function new(x) return setmetatable({ x = x }, V) end
local v = new(1)
print((v + v).x)
local holder = { item = v }
local function pick(a, b) if a then return a end return b end
local any = pick(holder, 5)
local ok, err = pcall(function() return any.item + 10 end)
print(ok, (tostring(err):gsub("^.-:%d+: ", "")))
