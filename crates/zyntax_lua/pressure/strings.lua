-- A string built, measured and dropped every step; the accumulator is
-- an integer so nothing keeps them.
local function main(n)
  local total = 0
  for i = 1, n do
    local s = "step " .. i
    local t = s .. "!"
    total = total + #t
  end
  return total
end

print(main(tonumber(arg[1])))
