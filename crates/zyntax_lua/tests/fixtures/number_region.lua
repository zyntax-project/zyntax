-- The loop's number is an integer on one path and a float on the
-- other, so the outlined region takes a Number aggregate as a live-in.
-- The inner loop runs long enough that the frame leaves through its
-- header rather than the outer loop's, and the step count arrives as
-- an argument: a function whose loop counts are known is compiled
-- before its first call and never leaves the interpreter at all.
local function main(count)
  local total = 0
  for n = 1, 3 do
    local x = n + 6
    local steps = 0
    while steps < count do
      if x % 2 == 0 then
        x = x / 2
      else
        x = 3 * x + 1
      end
      steps = steps + 1
    end
    total = total + steps + x
  end
  return total
end

print(main(20000000))
