-- The method is bound before the arguments run: an argument that
-- replaces or clears it does not change what this call calls.
local T = {}; T.__index = T
local function make(n) return function(self, x) return n end end
T.m = make(1)
local o = setmetatable({}, T)
local function swap() T.m = make(2); return 0 end
print(o:m(swap()))
print(o:m(0))
local function clear() T.m = nil; return 0 end
T.m = make(1)
print(o:m(clear()))
print(pcall(function() return o:m(0) end))
