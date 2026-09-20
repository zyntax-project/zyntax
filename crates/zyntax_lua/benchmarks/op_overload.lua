-- Operator overloading: a three-field vector with __add, two additions
-- an iteration, ten million iterations. Every addition allocates a
-- table, as the ZynML value-struct kernel does not.
-- Returns 210000000.

local Vec3 = {}
Vec3.__index = Vec3
Vec3.__add = function (a, b)
  return setmetatable({x = a.x + b.x, y = a.y + b.y, z = a.z + b.z}, Vec3)
end

local function new(x, y, z)
  return setmetatable({x = x, y = y, z = z}, Vec3)
end

local function main()
  local a = new(1.0, 2.0, 3.0)
  local b = new(4.0, 5.0, 6.0)
  local acc = new(0.0, 0.0, 0.0)
  for _ = 1, 10000000 do
    acc = acc + a
    acc = acc + b
  end
  return math.floor(acc.x + acc.y + acc.z)
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
