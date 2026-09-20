-- Free-function call latency, against inlined_call: the loop body is
-- one call to a local function doing the arithmetic the baseline does
-- inline, so the delta is the per-call cost that survives the pipeline.
-- Returns 350000000.

local function step(acc, i)
  return acc + i % 8
end

local function main()
  local sum = 0
  for i = 0, 99999999 do
    sum = step(sum, i)
  end
  return sum
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
