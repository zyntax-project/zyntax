-- A coroutine made, run to its first yield and dropped every step.
local function main(n)
  local total = 0
  for i = 1, n do
    local co = coroutine.wrap(function(a)
      coroutine.yield(a + 1)
      return a
    end)
    total = total + co(i)
  end
  return total
end

print(main(tonumber(arg[1])))
