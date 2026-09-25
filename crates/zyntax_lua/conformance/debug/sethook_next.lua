-- Setting or clearing a hook leaves table traversal alone.
local t = {a = 1}
debug.sethook()
print(next(t))

local Point = {}
Point.__index = Point
local p = setmetatable({x = 1, y = 2}, Point)
debug.sethook(function () end, "", 3)
local keys = {}
for k, v in pairs(p) do keys[#keys + 1] = k .. "=" .. v end
table.sort(keys)
print(table.concat(keys, " "))
debug.sethook()

-- A return hook that traverses a table on every return.
local seen = 0
local function aux ()
  local u = {b = 2}
  local k, v = next(u)
  if k == "b" and v == 2 then seen = seen + 1 end
end
debug.sethook(aux, "r")
local function f () return 1 end
f(); f()
debug.sethook()
print(seen > 0)
print(debug.gethook())
