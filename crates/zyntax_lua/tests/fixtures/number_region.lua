-- The loop's number is an integer on one path and a float on the
-- other, so the outlined region takes a Number aggregate as a live-in.
-- The bound arrives as an argument: a function whose loop count is
-- known is compiled before its first call, and never leaves the
-- interpreter through a region.
local function main(count)
  local total = 0
  for n = 1, count do
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

print(main(150000))
