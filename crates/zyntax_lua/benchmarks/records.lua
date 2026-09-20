-- Records as tables: rows built from constructors, filtered on a
-- field, aggregated by a key, then scanned for the best total. The
-- commonest shape of a data script, leaning on field access and on
-- integer keys in a hash. Returns 5997822.

local function main()
  local rows = {}
  for i = 0, 599999 do
    rows[#rows + 1] = {id = i, group = i % 17, score = (i * 7919) % 1000, flag = i % 3 == 0}
  end
  local totals = {}
  local count = 0
  for _, row in ipairs(rows) do
    if row.flag and row.score > 100 then
      local g = row.group
      totals[g] = (totals[g] or 0) + row.score
      count = count + 1
    end
  end
  local best = 0
  for _, total in pairs(totals) do
    if total > best then best = total end
  end
  return best + count
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
