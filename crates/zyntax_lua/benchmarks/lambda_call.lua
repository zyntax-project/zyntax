-- Calling a function value held in a local, nothing captured: ten
-- million iterations, as the ZynML and Python kernels of the name.
-- Returns 35000000.

local function main()
  local step = function (acc, i) return acc + i % 8 end
  local sum = 0
  for i = 0, 9999999 do
    sum = step(sum, i)
  end
  return sum
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
