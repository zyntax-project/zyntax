-- Fields holding an integer or a float, reached through the library:
-- pairs, next, rawget, rawset and a metatable's __index.
local Vec = {}
Vec.__index = Vec

function Vec.new(x, y)
  return setmetatable({ x = x, y = y }, Vec)
end

function Vec:len2()
  return self.x * self.x + self.y * self.y
end

local function fields(t)
  local out = {}
  for k, v in pairs(t) do
    out[#out + 1] = k .. "=" .. tostring(v) .. ":" .. math.type(v)
  end
  table.sort(out)
  return table.concat(out, " ")
end

local a = Vec.new(3, 4.5)
local b = Vec.new(1.5, 2)
print(fields(a), fields(b))
print(a:len2(), b:len2())

-- next on a table with one field, both kinds
local one = { v = 1 }
one.v = one.v + 0.5
print(next(one))
one.v = 2
print(next(one))
print(next(one, "v"))

-- rawget and rawset go through the slot
print(rawget(a, "x"), math.type(rawget(a, "x")), rawget(a, "y"), math.type(rawget(a, "y")))
rawset(a, "x", 2.25)
rawset(a, "y", 8)
print(a.x, math.type(a.x), a.y, math.type(a.y), a:len2())
rawset(a, "x", 5)
print(a.x, math.type(a.x), a.x // 2, a.x / 2, fields(a))

-- __index of a metatable reading a Number field of another table
local defaults = Vec.new(10, 0.5)
local p = setmetatable({}, { __index = defaults })
print(p.x, math.type(p.x), p.y, math.type(p.y))
defaults.x = 10.0
defaults.y = 1
print(p.x, math.type(p.x), p.y, math.type(p.y))
local q = setmetatable({}, { __index = function(_, k) return rawget(defaults, k) end })
print(q.x, math.type(q.x), q.y, math.type(q.y))

-- string.format and concatenation keep each kind's text
local d = Vec.new(2 ^ 53, -0.0)
print(string.format("%.0f %g %s %s", d.x, d.x, d.x, d.y), d.x .. "|" .. d.y, 1 / d.y)
d.x = -9223372036854775807 - 1
d.y = -7
print(d.x, d.y, d.x .. "|" .. d.y, math.type(d.x // d.y))
