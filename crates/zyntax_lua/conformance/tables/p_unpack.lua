-- what table.unpack hands out of a table is a value like any other

local rows = {}
for i = 1, 3 do rows[i] = { v = i } end
local r = table.unpack(rows)
r.v = 2.5
print(rows[1].v + 1)

local a, b, c = table.unpack(rows)
c.v = "three"
print(a.v + 1, b.v, rows[3].v, #rows)

local nums = { 1, 2, 3 }
print(table.unpack(nums))
print(select("#", table.unpack(nums)), table.concat(nums, ","), rawlen(nums))
