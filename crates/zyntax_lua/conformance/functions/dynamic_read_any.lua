-- a function read out of a table under a key not known at compile time
-- is called with whatever the caller passes

local ops = { inc = function(x) return x + 1 end }
print(ops.inc(1), ops.inc(2))
local name = "in" .. "c"
local f = ops[name]
print(f("10"), f(1.5))
local g = rawget(ops, name)
print(g("20"))

-- through a class's __index
local Class = { v = 1 }
Class.__index = Class
function Class.get(self, k) return self.v + k end
local o = setmetatable({}, Class)
print(o:get(1))
local method = "g" .. "et"
print(o[method](o, "2"), o[method](o, 0.5))

-- through a receiver the types do not follow
local handlers = { double = function(x) return x * 2 end }
print(handlers.double(4))
local function pick(a, b) if a then return a end return b end
local any = pick(handlers, 0)
local which = "dou" .. "ble"
print(any[which]("3"), any[which](1.25))

-- through a metatable the types do not follow
local Hidden = { secret = function(x) return x + 1 end }
print(Hidden.secret(1))
local meta = pick({ __index = Hidden }, 0)
local obj = setmetatable({ z = 0 }, meta)
local key = "sec" .. "ret"
print(obj[key]("5"), obj[key](0.25))
