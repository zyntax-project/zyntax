-- A function called as a field of a class table.
local Point = {}
Point.__index = Point
function Point.new(x, y) return setmetatable({x = x, y = y}, Point) end
function Point.sum(p) return p.x + p.y end
local total = 0
for i = 1, 3 do
  local p = Point.new(i, i)
  total = total + Point.sum(p)
end
print(total)
