-- what select hands out of its arguments is a value like any other

local rows = {}
for i = 1, 3 do rows[i] = { v = i } end
local r = select(2, table.unpack(rows))
r.v = 1.5
print(rows[2].v + 2)

local s = select(1, rows)
s[3].v = 0.25
print(rows[3].v * 4, select("#", rows, rows[1]), select(-1, 1, 2, rows[1].v))
