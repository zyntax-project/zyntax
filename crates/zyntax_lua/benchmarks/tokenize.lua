-- Text handling: strings built from pieces, joined, split back into
-- tokens, counted into a table keyed by string, and formatted out
-- again. String allocation, lookups by string key and tostring are
-- the whole of it. Returns 2729.

local function main()
  local words = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
  local parts = {}
  for i = 0, 299999 do
    parts[#parts + 1] = words[i % 8 + 1] .. tostring(i % 100)
  end
  local text = table.concat(parts, " ")
  local counts = {}
  for tok in string.gmatch(text, "[^ ]+") do
    counts[tok] = (counts[tok] or 0) + 1
  end
  local out = {}
  local n = 0
  for key, c in pairs(counts) do
    n = n + 1
    if c > 350 then out[#out + 1] = key .. "=" .. tostring(c) end
  end
  table.sort(out)
  local joined = table.concat(out, ",")
  return #joined + n
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
