-- what rawget finds in a table is the table's own element, typed or not

local rows = {}
for i = 1, 3 do rows[i] = { v = i } end
local r = rawget(rows, 1)
r.v = 2.5
print(rows[1].v + 1)

local named = { first = { v = 1 }, n = 2 }
local f = rawget(named, "first")
f.v = "one"
print(named.first.v, rawget(named, "n") + 1, rawget(named, "none"))

local k = "first"
local g = rawget(named, k)
g.v = 0.5
print(named.first.v * 2)
