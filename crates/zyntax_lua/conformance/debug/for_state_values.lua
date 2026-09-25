-- The hidden locals of a for loop, as debug.getlocal sees them.
local function show (lv)
  for i = 1, 20 do
    local n, v = debug.getlocal(lv + 1, i)
    if not n then break end
    print(i, n, type(v))
  end
end
local x = 1
for k, v in pairs({10}) do show(1) break end
for i = 1, 2 do show(1) break end

local function values (lv)
  local out = {}
  for i = 1, 20 do
    local n, v = debug.getlocal(lv + 1, i)
    if not n then break end
    if n == "(for state)" then out[#out + 1] = tostring(v) end
  end
  return table.concat(out, " ")
end
for i = 1, 3 do print("numeric", values(1)) end
for i = 10, 1, -4 do print("down", values(1)) end
for i = 1.5, 2.5, 0.5 do print("float", values(1)) end
local s = 2
for i = 1, s do print("dynamic", values(1)) end
for i, v in ipairs({"a", "b"}) do
  local n1, f = debug.getlocal(1, 5)
  local n2, t = debug.getlocal(1, 6)
  local n3, c = debug.getlocal(1, 7)
  local n4, z = debug.getlocal(1, 8)
  print("ipairs", n1, type(f), type(t), c, z)
end
local c <const> = nil
local closing = setmetatable({}, {__close = function () end})
for k in next, {5}, nil, closing do
  local n, v = debug.getlocal(1, 9)
  print("generic closing", n, v == closing)
end
