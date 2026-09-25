-- A collection in the middle of a traversal of a weak table only
-- clears fields: every entry still held is visited once. A new key
-- stored afterwards finds the table as the collection left it.
local function scrub(n)
  local a, b, c, d, e = n, n, n, n, n
  if n > 0 then return scrub(n - 1) + a end
  return 0
end

local function fill(t, held)
  for i = 1, 200 do
    local k = {id = i}
    t[k] = i
    if i % 2 == 0 then held[#held + 1] = k end
    t[{}] = -i
  end
end

local function visit(seen, k, v, steps)
  local dup = 0
  if v > 0 then
    if seen[v] then dup = 1 end
    seen[v] = true
  end
  if steps == 10 then scrub(200); collectgarbage() end
  return dup
end

local function walk(t, form)
  local seen, dup, steps = {}, 0, 0
  if form == "pairs" then
    for k, v in pairs(t) do
      steps = steps + 1
      dup = dup + visit(seen, k, v, steps)
    end
  else
    for k, v in next, t do
      steps = steps + 1
      dup = dup + visit(seen, k, v, steps)
    end
  end
  return seen, dup
end

local function check(mode, form)
  local t = setmetatable({}, {__mode = mode})
  local held = {}
  fill(t, held)
  local seen, dup = walk(t, form)
  local missing = 0
  for _, k in ipairs(held) do
    if not seen[k.id] then missing = missing + 1 end
  end
  -- New keys after the traversal, then every held key looked up.
  for i = 1, 50 do t["n" .. i] = i end
  local found, count = 0, 0
  for _, k in ipairs(held) do
    if t[k] == k.id then found = found + 1 end
  end
  for k, v in pairs(t) do count = count + 1 end
  print(mode, form, "dup", dup, "missing", missing, "found", found,
    "few left", count < 200)
end

check("k", "pairs")
check("k", "next")
check("kv", "pairs")
check("kv", "next")
