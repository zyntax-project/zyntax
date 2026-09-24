-- what pairs and next hand out of a table is a value like any other

local t = { add = function(a, b) return a + b end, n = 1 }
print(t.add(1, 2))
for k, v in pairs(t) do
  if type(v) == "function" then print(k, v("1", "2")) end
end
local key, value = next(t)
while key do
  if type(value) == "function" then print(key, value(0.5, "1")) end
  key, value = next(t, key)
end

local inner = { point = { x = 1 } }
print(inner.point.x + 1)
for _, v in pairs(inner) do v.x = "one" end
print(inner.point.x)

-- a table the types do not follow, walked
local scale = { by = function(x) return x * 3 end }
print(scale.by(2))
local function pick(a, b) if a then return a end return b end
local blind = pick(scale, false)
for _, fn in pairs(blind) do print(fn("5")) end

-- the library's walkers held as values
local held = { f = function(x) return x + 1 end }
print(held.f(1))
for k, v in next, held do print(k, v("2")) end
local it, st = pairs(held)
local key, fn = it(st)
print(key, fn(0.5))
local unpack = table.unpack
local list = { function(x) return x * 2 end }
print(list[1](2))
local first = unpack(list)
print(first("3"))
for k, v in pairs{ h = function(x) return x - 1 end } do print(k, v("9")) end
