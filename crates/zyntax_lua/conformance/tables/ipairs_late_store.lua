-- a loop over ipairs binds what the table holds by the end of the
-- program, not what it held where the loop is written

local t = { 1, 2 }
local function show()
  for _, v in ipairs(t) do
    if v then print(v) end
  end
end
local function order()
  table.sort(t, function(a, b) return tostring(a) < tostring(b) end)
end
t[3] = "x"
show()
order()
show()

local src = setmetatable({}, { __index = { 5, "six" } })
local dst = { 1, 2 }
table.move(src, 1, 2, 1, dst)
print(dst[1], dst[2])
