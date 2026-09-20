-- A pipeline of tables: a list built by a loop, filtered into another,
-- reduced, paired, sorted by a key and unpacked. Every stage allocates
-- or calls back. The sort's order is total, so the pairs taken after it
-- are the same whatever the sort. Returns 817368200.

local function main()
  local n = 1500000
  local xs = {}
  for i = 0, n - 1 do
    xs[i + 1] = (i * 2654435761) % 1000003
  end
  local evens = {}
  for i = 1, n do
    local x = xs[i]
    if x % 2 == 0 then evens[#evens + 1] = x end
  end
  local squares = 0
  for i = 1, #evens do
    local x = evens[i]
    squares = squares + x * x % 1000
  end
  local pairs_ = {}
  for i = 1, 200000 do
    local x = evens[i]
    pairs_[i] = {x % 1000, x}
  end
  table.sort(pairs_, function (p, q)
    if p[1] ~= q[1] then return p[1] < q[1] end
    return p[2] < q[2]
  end)
  local acc = squares
  for i = 1, 1000 do
    local p = pairs_[i]
    acc = acc + p[1] * 31 + p[2]
  end
  return acc % 1000000007
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
