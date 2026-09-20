-- Collatz step-count sum: a data-dependent trip count, an unpredictable
-- branch, nothing loop-invariant, a strictly sequential recurrence.
-- What is left is branch layout, register allocation and scalar
-- simplification. The halving is a division, which LuaJIT's Lua 5.1
-- syntax allows where `//` is not; under 5.4 the value is then a float.
-- Returns 35669673.

local function main()
  local total = 0
  for n = 1, 299999 do
    local x = n
    local steps = 0
    while x > 1 do
      if x % 2 == 0 then
        x = x / 2
      else
        x = 3 * x + 1
      end
      steps = steps + 1
    end
    total = total + steps
  end
  return total
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
