-- __index handlers: what they return, where they sit, and when they
-- are replaced.
local calls = 0
local P = {}
P.__index = function(t, k)
  calls = calls + 1
  if k == "get" then return function(self) return self.x end end
  if k == "name" then return "a string" end
  return nil
end
local p = setmetatable({x = 5}, P)
print(p:get(), calls)
print(p.name, calls)
print(pcall(function() return p:name() end))
print(pcall(function() return p:nothing() end))
local f = p.get
print(f(p), calls)
-- a handler on the class's class
local Base = setmetatable({}, {__index = function(t, k)
  return function(self) return "base " .. k end
end})
Base.__index = Base
local q = setmetatable({}, Base)
print(q:hello(), q:bye())
function Base:hello() return "own hello" end
print(q:hello(), q:bye())
-- the handler replaced by a table at run time
local R = {}
R.__index = function(t, k) return function() return "fn " .. k end end
local r = setmetatable({}, R)
print(r:go())
R.__index = {go = function() return "table go" end}
print(r:go())
-- an own field shadows the handler
local s = setmetatable({}, P)
s.get = function(self) return "own" end
print(s:get())
s.get = nil
s.x = 9
print(s:get())
-- rawset then read
rawset(s, "get", function() return "raw" end)
print(s:get(), s.get(s))
print(calls)
-- the handler runs before the arguments
local log = {}
local L = {}
L.__index = function(t, k)
  log[#log + 1] = "index " .. k
  return function(self, x) return "got " .. tostring(x) end
end
local l = setmetatable({}, L)
local function arg() log[#log + 1] = "arg"; return 1 end
print(l:go(arg()))
print(table.concat(log, ","))
