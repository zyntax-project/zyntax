-- Data-dependent multi-way dispatch: the branch taken each iteration
-- depends on the running value, so the chain cannot be flattened or
-- hoisted. Branch ordering and register pressure, not loop shape.
-- Returns what the ZynML kernel returns.

local function main()
  local acc = 1
  for _ = 1, 20000000 do
    local k = acc % 7
    if k == 0 then
      acc = acc + 3
    elseif k == 1 then
      acc = acc * 2 + 1
    elseif k == 2 then
      acc = acc - 5
    elseif k == 3 then
      acc = acc + 11
    elseif k == 4 then
      acc = acc * 3
    elseif k == 5 then
      acc = acc + 7
    else
      acc = acc - 1
    end
    acc = acc % 1000003
  end
  return acc
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
