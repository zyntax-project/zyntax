-- a table read out under a key not known at compile time, or by the
-- library, takes stores the types do not see

local obj = { pos = { x = 1 } }
local k = "po" .. "s"
local p = obj[k]
p.y = 2.5
local y = obj.pos.y
print(y)
if y then print(y + 1) end
local r = obj.pos.x
p.x = "s"
print(obj.pos.x, r + 1)

local rows = { { v = 1 }, { v = 2 } }
local a = table.unpack(rows)
a.v = 2.5
print(rows[1].v + 1)

local cells = { { n = 1 }, { n = 2 } }
local last = table.remove(cells)
last.n = "two"
local kept = { last }
print(kept[1].n, cells[1].n + 1)
