-- Stores a class's metatable swallows never reach the class.
local C = setmetatable({}, {__newindex = function() end})
C.__index = C
function C:get() return 1 end
local o = setmetatable({}, C)
print(rawget(C, "get"), rawget(C, "__index"))
print(pcall(function() return o:get() end))
