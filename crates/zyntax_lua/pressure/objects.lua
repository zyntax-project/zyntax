-- One object made and dropped per step: the operator's result replaces
-- the accumulator.
local Vec = {}
Vec.__index = Vec
Vec.__add = function(a, b) return Vec.new(a.x + b.x) end

function Vec.new(x)
  return setmetatable({ x = x }, Vec)
end

local function main(n)
  local a = Vec.new(1.0)
  local acc = Vec.new(0.0)
  for i = 1, n do
    acc = acc + a
  end
  return math.floor(acc.x)
end

print(main(tonumber(arg[1])))
