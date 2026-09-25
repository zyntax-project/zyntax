-- Each closure instance over a fresh variable has its own upvalue
-- identity; closures over one variable share it.
local function mk ()
  local v = 0
  return function () v = v + 1; return v end
end
local f1, f2 = mk(), mk()
print(debug.upvalueid(f1, 1) == debug.upvalueid(f2, 1))
print(debug.upvalueid(f1, 1) == debug.upvalueid(f1, 1))

local function pair ()
  local s = 0
  return function () return s end, function (x) s = x end
end
local get, set = pair()
print(debug.upvalueid(get, 1) == debug.upvalueid(set, 1))

local function const ()
  local c = 42
  return function () return c end
end
print(debug.upvalueid(const(), 1) == debug.upvalueid(const(), 1))

-- Closures made across the iterations of a loop with a goto, as the
-- official goto test builds them.
local fs = {}
do
  local x = 1
  goto l1
  ::l2::
  do return end
  ::l1::
  local y = 2
  local i = 0
  ::again::
  i = i + 1
  local z = i
  fs[#fs + 1] = function () return x, y, z end
  if i < 3 then goto again end
end
for i = 2, 3 do
  print(i,
    debug.upvalueid(fs[1], 1) == debug.upvalueid(fs[i], 1),
    debug.upvalueid(fs[1], 2) == debug.upvalueid(fs[i], 2),
    debug.upvalueid(fs[1], 3) == debug.upvalueid(fs[i], 3))
end
