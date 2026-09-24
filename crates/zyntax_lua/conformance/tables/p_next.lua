-- what next and pairs hand out of a table is typed as everything the
-- table holds, or a value like any other

local rows = {}
for i = 1, 3 do rows[i] = { v = i } end
local _, r = next(rows)
r.v = 2.5
print(rows[1].v + 1)

-- a table with an array part and fields of other kinds
local mixed = { 10, 20, 30, name = "m", scale = 0.5 }
local ints, strs, floats = 0, 0, 0
for k, v in pairs(mixed) do
  if math.type(v) == "integer" then ints = ints + v
  elseif type(v) == "string" then strs = strs + #v + #k
  else floats = floats + v end
end
print(ints, strs, floats)

-- a walk that stores into what it finds
local nested = { { n = 1 }, { n = 2 } }
for _, item in pairs(nested) do item.n = item.n + 0.5 end
print(nested[1].n, nested[2].n)
for _, item in ipairs(nested) do item.n = "x" .. item.n end
print(nested[1].n, nested[2].n)

-- counts by a key read from a string
local counts = {}
for w in string.gmatch("a b a c a b", "%a") do
  counts[w] = (counts[w] or 0) + 1
end
local total = 0
for _, c in pairs(counts) do total = total + c end
print(counts.a, counts.b, counts.c, counts.d, total)
