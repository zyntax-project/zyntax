-- The inlined baseline of the call-latency pair: the arithmetic of
-- free_function_call spelled in the loop body. The accumulator steps by
-- i % 8 so the loop is not an affine recurrence with a closed form.
-- Returns 350000000.

local function main()
  local sum = 0
  for i = 0, 99999999 do
    sum = sum + i % 8
  end
  return sum
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
