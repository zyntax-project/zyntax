-- A driver loop in a function that returns the table it fills: the
-- interpreted frame leaves through a resume point, whose result is the
-- table, and the second call runs the compiled body.
local function main(n)
  local times = {}
  local total = 0
  for i = 0, n - 1 do
    total = total + ((i * 7) & 1023)
    if i % 100000 == 0 then
      times[#times + 1] = total
    end
  end
  return times
end

local r = main(3000000)
print(#r, r[1], r[#r])
local r2 = main(3000000)
print(#r2, r2[#r2])
