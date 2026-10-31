-- Instances of two classes share a shape, and one loses its metatable:
-- each operation finds its handler by the metatable it meets.
local V, W = {}, {}
V.__add = function(p, q) return setmetatable({x = p.x + q.x}, V) end
W.__add = function(p, q) return setmetatable({x = -1}, W) end
local a = setmetatable({x = 1}, V)
local c = setmetatable({x = 5}, W)
print((a + c).x, (c + a).x, (a + a).x)
local d = setmetatable({x = 7}, V)
setmetatable(d, nil)
print(pcall(function() return d + d end))
print((d + a).x)
