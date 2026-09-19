-- Metatables: __index classes, arithmetic and comparison metamethods,
-- __tostring, __len, __call, __newindex, inheritance.
local Point = {}
Point.__index = Point

function Point.new(x, y)
  return setmetatable({ x = x, y = y }, Point)
end

function Point:add(o)
  return Point.new(self.x + o.x, self.y + o.y)
end

function Point.__add(a, b)
  return Point.new(a.x + b.x, a.y + b.y)
end

function Point.__eq(a, b)
  return a.x == b.x and a.y == b.y
end

function Point.__lt(a, b)
  return a.x < b.x or (a.x == b.x and a.y < b.y)
end

function Point.__le(a, b)
  return not (b < a)
end

function Point.__tostring(p)
  return "(" .. p.x .. ", " .. p.y .. ")"
end

function Point.__len(p)
  return 2
end

function Point:norm2()
  return self.x * self.x + self.y * self.y
end

local a = Point.new(1, 2)
local b = Point.new(3, 4)
print(tostring(a), tostring(a:add(b)), tostring(a + b))
print(a == b, a == Point.new(1, 2), a < b, a <= b, b < a, a ~= b)
print(#a, a:norm2(), b:norm2())
print(getmetatable(a) == Point, getmetatable(1))

local Animal = {}
Animal.__index = Animal
function Animal.new(name, sound)
  local self = setmetatable({}, Animal)
  self.name = name
  self.sound = sound
  return self
end
function Animal:speak() return self.name .. " says " .. self.sound end
function Animal:kind() return "animal" end

local Dog = setmetatable({}, { __index = Animal })
Dog.__index = Dog
function Dog.new(name)
  local self = Animal.new(name, "woof")
  return setmetatable(self, Dog)
end
function Dog:kind() return "dog" end
local d = Dog.new("Rex")
print(d:speak(), d:kind(), Animal.new("Cat", "meow"):kind())

local defaults = setmetatable({}, { __index = function(t, k) return "default:" .. k end })
print(defaults.anything, defaults[42])

local log = {}
local guarded = setmetatable({}, {
  __newindex = function(t, k, v)
    log[#log + 1] = k
    rawset(t, k, v)
  end,
})
guarded.a = 1
guarded.b = 2
guarded.a = 3
print(#log, log[1], log[2], guarded.a, guarded.b)

local callable = setmetatable({}, { __call = function(self, x, y) return x * y end })
print(callable(6, 7))

local V = setmetatable({}, { __index = function(_, k) return k * 2 end })
print(V[21])
local counter = setmetatable({ n = 0 }, { __call = function(self) self.n = self.n + 1 return self.n end })
counter() counter()
print(counter(), counter.n)
local mt = { __concat = function(a, b) return "cat" end, __unm = function() return "neg" end }
local c = setmetatable({}, mt)
print(c .. "x", "x" .. c, -c)
local proxy = setmetatable({}, { __index = a })
print(proxy.x, proxy.y, rawget(proxy, "x"))
