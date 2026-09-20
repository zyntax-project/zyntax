-- table.concat over a list of pieces every step: the pieces, the list
-- and the joined string are all dropped by the next.
local function main(n)
  local total = 0
  for i = 1, n do
    local parts = {}
    for k = 1, 8 do parts[k] = "p" .. (i + k) end
    total = total + #table.concat(parts, ",")
  end
  return total
end

print(main(tonumber(arg[1])))
