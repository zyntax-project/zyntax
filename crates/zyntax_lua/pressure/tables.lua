-- An array made, grown, read and dropped every step.
local function main(n)
  local total = 0
  for i = 1, n do
    local xs = { i, i + 1, i + 2 }
    xs[#xs + 1] = i + 3
    total = total + xs[4] - xs[1]
  end
  return total
end

print(main(tonumber(arg[1])))
