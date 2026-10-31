-- An arithmetic handler stored before an operation is called where the
-- operation stands; one stored after it is still looked up.
local V = {}
V.__index = V
local function new(x) return setmetatable({x = x}, V) end
local a, b = new(1), new(2)
print(pcall(function() return a + b end))
V.__add = function(p, q) return new(p.x + q.x) end
print((a + b).x)
local acc = new(0)
for _ = 1, 10 do acc = acc + a; acc = acc + b end
print(acc.x)
print(pcall(function() return acc - b end))
V.__sub = function(p, q) return new(p.x - q.x) end
print((acc - b).x)
