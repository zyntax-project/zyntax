-- A coroutine made, run to its end and dropped every step.
local function main(n)
  local total = 0
  for i = 1, n do
    local co = coroutine.wrap(function(a)
      local b = coroutine.yield(a + 1)
      return b + 1
    end)
    total = total + co(i) + co(i)
  end
  return total
end

print(main(tonumber(arg[1])))
