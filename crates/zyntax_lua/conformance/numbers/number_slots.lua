-- A table field that holds an integer at one time and a float at
-- another keeps each value's kind.
local Point = {}
Point.__index = Point

function Point.new(x, y)
  return setmetatable({ x = x, y = y }, Point)
end

function Point:shift(d)
  self.x = self.x + d
  self.y = self.y * 2
end

local function show(label, v)
  print(label, v, math.type(v), tostring(v), string.format("%q", v))
end

local p = Point.new(1, 2.5)
local q = Point.new(0.5, 3)
show("p.x", p.x)
show("p.y", p.y)
show("q.x", q.x)
show("q.y", q.y)

p:shift(1)
q:shift(1)
show("p.x", p.x)
show("p.y", p.y)
show("q.x", q.x)
show("q.y", q.y)

-- The same field, integer then float then integer.
p.x = 7
show("int", p.x)
p.x = 7.0
show("float", p.x)
print(p.x == 7, rawequal(p.x, 7), math.type(p.x))
p.x = 7
show("int again", p.x)

-- Floor division and modulo by zero: an error for integers, inf or
-- nan for floats.
p.x = 5
print(pcall(function() return p.x // 0 end))
print(pcall(function() return p.x % 0 end))
p.x = 5.0
print(p.x // 0, -p.x // 0, p.x % 0 ~= p.x % 0)

-- A numeric for loop from an integer field counts in integers.
p.x = 1
for i = p.x, 2 do io.write(math.type(i), " ", tostring(i), " ") end
print()

-- Extremes round trip exactly.
p.x = -9223372036854775807 - 1
print(p.x, p.x == math.mininteger, math.type(p.x))
p.x = 9223372036854775807
print(p.x, p.x + 1 == math.mininteger)
p.x = 2 ^ 63
print(p.x, math.type(p.x), p.x == 2 ^ 63, string.format("%.1f", p.x))
p.x = -0.0
print(p.x, 1 / p.x, tostring(p.x))
p.x = 0
print(p.x, 1 / p.x, tostring(p.x))

-- Arithmetic mixing the field's two kinds.
local a = Point.new(3, 0.25)
local b = Point.new(2.0, 4)
print(a.x + b.x, a.x - b.y, a.x * b.y, a.y * b.y, a.x / b.y, a.x // b.y, a.x % b.y)
print(a.x < b.x, a.x <= b.y, a.y < b.y, a.x == 3.0, b.x == 2, a.y > b.y)
print(math.maxinteger + 0.0 == math.maxinteger, math.maxinteger < 2 ^ 63)
a.x = 9223372036854775807
b.x = 2.0 ^ 63
print(a.x < b.x, a.x == b.x, b.x > a.x, a.x + 1 < b.x)
a.x = 2 ^ 53 + 1
b.x = 2.0 ^ 53
print(a.x == b.x, a.x > b.x, b.x < a.x, a.x + 0.0 == b.x)

-- Sums over a table of such points.
local pts = {}
for i = 1, 6 do
  if i % 2 == 0 then
    pts[i] = Point.new(i, i / 2)
  else
    pts[i] = Point.new(i + 0.5, i)
  end
end
local sx, sy = 0, 0
for _, pt in ipairs(pts) do
  sx = sx + pt.x
  sy = sy + pt.y
end
print(sx, sy, math.type(sx), math.type(sy))
