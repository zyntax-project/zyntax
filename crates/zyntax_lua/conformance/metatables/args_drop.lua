-- A method cleared by its own argument still runs once.
local T = {}; T.__index = T
function T:m(x) return "m " .. tostring(x) end
local o = setmetatable({}, T)
print(o:m((function() T.m = nil; return 1 end)()))
print(pcall(function() return o:m(1) end))
-- The same for a method held in the instance itself.
local own = {}
own.m = function(self, x) return "own " .. tostring(x) end
print(own:m((function() own.m = nil; return 2 end)()))
print(pcall(function() return own:m(2) end))
