-- A record with string keys made, written, read and dropped every step.
local function main(n)
  local total = 0
  for i = 1, n do
    local d = { a = i, b = i + 1 }
    d.c = i + 2
    total = total + d.c - d.a
  end
  return total
end

print(main(tonumber(arg[1])))
