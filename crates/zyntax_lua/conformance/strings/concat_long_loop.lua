-- Strings built by concatenation in a loop long enough to tier up,
-- every piece kept in a table: none is lost and none is nil.
local function build(n)
  local words = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
  local parts = {}
  for i = 0, n - 1 do
    parts[#parts + 1] = words[i % 8 + 1] .. tostring(i % 100)
  end
  return parts
end

local parts = build(300000)
print(#parts, parts[1], parts[150000], parts[300000])
print(#table.concat(parts, " "))

local counts = {a = 1000, b = 1000, c = 1000}
local out = {}
for key, c in pairs(counts) do
  if c > 350 then out[#out + 1] = key .. "=" .. tostring(c) end
end
table.sort(out)
print(#out, table.concat(out, ","))
