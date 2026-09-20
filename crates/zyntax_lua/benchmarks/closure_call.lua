-- Calling a closure that captures a variable of its defining function:
-- the captured k is read on every call, on top of what lambda_call
-- pays. Ten million iterations. Returns 35000000.

local function make_step(k)
  return function (acc, i)
    return acc + (i + k) % 8
  end
end

local function main()
  local step = make_step(3)
  local total = 0
  for i = 0, 9999999 do
    total = step(total, i)
  end
  return total
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
