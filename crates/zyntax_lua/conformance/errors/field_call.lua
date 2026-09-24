-- A nil field or method called: the message names which it was.
local Class = {}
Class.__index = Class
print(pcall(function() return Class.new() end))
local o = setmetatable({}, Class)
print(pcall(function() return o:new() end))
print(pcall(function() return o.new() end))
function Class.new() return setmetatable({}, Class) end
print(pcall(function() return Class.new() ~= nil end))
print(pcall(function() return o:new() ~= nil end))
local t = {inner = {}}
print(pcall(function() return t.inner.fn() end))
print(pcall(function() return t.inner:fn() end))
print(pcall(function() local u = nil; return u:m() end))
