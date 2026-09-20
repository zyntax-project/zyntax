-- A closure over the loop variable made and called every step.
local function apply(f, v)
  return f(v)
end

local function main(n)
  local total = 0
  for i = 1, n do
    local f = function(x) return x + i end
    total = total + apply(f, 1)
  end
  return total
end

print(main(tonumber(arg[1])))
