-- A field holding any value, written and read back: a fresh table per
-- iteration holding one float, read out through the field. A table's
-- fields are dynamic, so this is the boxing round trip the ZynML and
-- Python kernels of the same name measure, plus a table allocation.
-- Returns 1500000.

local function main()
  local sum = 0.0
  for _ = 1, 1000000 do
    local bag = {payload = 1.5}
    local v = bag.payload
    sum = sum + v
  end
  return math.floor(sum)
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
