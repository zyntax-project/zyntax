local Pair = {}
Pair.__index = Pair
function Pair.new(x, y)
  return setmetatable({x = x, y = y}, Pair)
end

local function advance(n)
  local initial = Pair.new(1.5, 2.0)
  local current = initial
  for i = 1, n do
    current = Pair.new(current.x + current.y, current.y + 0.5)
  end
  return current.x, current.y, initial.x, initial.y
end

for _, n in ipairs({0, 1, 20000}) do
  print(advance(n))
end

local old = Pair.new(4.0, 5.0)
local current = old
for i = 1, 3 do
  current.x = current.x + 2.0
  current = Pair.new(current.x, current.y + 1.0)
end
print(old.x, old.y, current.x, current.y)
