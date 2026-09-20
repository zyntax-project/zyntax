-- What an uncallable value is named as: itself, not its arguments.
local T = {}; T.__index = T
local o = setmetatable({v = 1}, T)
print(pcall(function() return o.m(o) end))
print(pcall(function() return o.m(1) end))
print(pcall(function() return o:m() end))
print(pcall(function() return undefined_global(2, 3) end))
local n = 5
print(pcall(function() return n(o) end))
local callable = setmetatable({}, {__call = function(self, x) return x * 2 end})
print(callable(21))
local nested = setmetatable({}, {__call = function(self, x) return x + 1 end})
print(nested(4))
local twice = setmetatable({}, {__call = callable})
print(pcall(function() return twice(4) end))
local uncallable = setmetatable({}, {__call = 7})
print(pcall(function() return uncallable(1) end))
local plain = {}
print(pcall(function() return plain(1) end))
