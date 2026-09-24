-- Metatables set through setmetatable reached as a value, not called
-- by name: slot reads, reads the class chain answers, and lengths of
-- the tables it reaches follow the metatable.

local function mk(v) return {f = v, 1, 2, 3} end
mk(1)

-- 1. a local alias
local sm = setmetatable
local a = mk(nil)
sm(a, {__index = function() return 7 end})
print(1, a.f, a.g, rawget(a, "f"))

-- 2. through pcall, and its errors
local b = mk(nil)
print(2, (pcall(setmetatable, b, {__index = function(_, k) return "via pcall " .. k end})))
print(2, b.f)
print(2, pcall(setmetatable, {x = 1}, 5))
local guarded = setmetatable({x = 1}, {__metatable = "locked"})
print(2, getmetatable(guarded), pcall(setmetatable, guarded, {}))

-- 3. a class whose own metatable is set through an alias
local Base = {}
Base.__index = Base
local function obj(v) return setmetatable({f = v}, Base) end
local c = obj(nil)
obj(1)
print(3, c.f)
sm(Base, {__index = function() return 9 end})
print(3, c.f, c.h)

-- 4. a metatable given its __index after the alias sets it
local d = mk(nil)
local mt = {}
sm(d, mt)
print(4, d.f)
mt.__index = {f = 3}
print(4, d.f)

-- 5. a length handler set through an alias
local e = mk(1)
print(5, #e)
local len_key = string.char(95, 95, 108, 101, 110)
sm(e, {[len_key] = function() return 42 end})
print(5, #e)

-- 6. the function kept in a table, fetched from the globals table,
-- through the debug library taken as a value
local fns = {sm}
local f6 = mk(nil)
fns[1](f6, {__index = function() return 61 end})
local g6 = mk(nil)
_G.setmetatable(g6, {__index = function() return 62 end})
local h6 = mk(nil)
rawget(_G, "setmet" .. "atable")(h6, {__index = function() return 63 end})
local dbg = debug
local i6 = mk(nil)
dbg.setmetatable(i6, {__index = function() return 64 end})
print(6, f6.f, g6.f, h6.f, i6.f)

-- 7. set on a table the call's result is read from
local j = mk(nil)
local k7 = setmetatable(j, {__index = function() return 71 end}).f
print(7, k7, j.f)
