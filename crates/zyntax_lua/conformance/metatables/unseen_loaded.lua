-- Metatables set by code the program loads while it runs.

local function mk(v) return {f = v, g = 1, 10, 20} end
local o = mk(nil)
local p = mk(1)
print(1, o.f, p.f, #o)
Obj = o
load("setmetatable(Obj, {__index = function() return 5 end, __len = function() return 50 end})")()
print(2, o.f, p.f, #o)
local q = mk(nil)
local attach = load("return function(t) setmetatable(t, {__index = function() return 6 end}) end")()
attach(q)
print(3, q.f, q.g)
