-- fields absent from a table, looked up through its metatables

-- 1. a class chain that never holds the field (a tree's leaves)
local Tree = {}
Tree.__index = Tree
function Tree.new(item, depth)
  local self = setmetatable({}, Tree)
  self.item = item
  if depth > 0 then
    self.left = Tree.new(item * 2 - 1, depth - 1)
    self.right = Tree.new(item * 2, depth - 1)
  end
  return self
end
function Tree:check()
  if not self.left then return self.item end
  return self.item + self.left:check() - self.right:check()
end
local tree = Tree.new(1, 3)
print(tree:check(), tree.left.left.left.left, rawget(tree.left, "left") ~= nil)

-- 2. a chain that gains the field later, through a store to the class
local Base = {}
Base.__index = Base
local Kid = setmetatable({}, Base)
Kid.__index = Kid
local k = setmetatable({own = 1}, Kid)
local function peek(o) return o.extra end
print(peek(k), rawget(k, "extra"))
Base.extra = "from base"
print(peek(k), rawget(k, "extra"))
Kid.extra = "from kid"
print(peek(k))
k.extra = "own"
print(peek(k), rawget(k, "extra"))
k.extra = nil
print(peek(k))

-- 3. rawget against the chain
local Cls = {shared = 10}
Cls.__index = Cls
local o = setmetatable({mine = 1}, Cls)
print(o.shared, rawget(o, "shared"), o.mine, rawget(o, "mine"))

-- 4. a chain ending in an __index function
local Top = {}
Top.__index = function(t, key) return "computed " .. key end
local Mid = setmetatable({}, Top)
Mid.__index = Mid
local m = setmetatable({a = 1}, Mid)
print(m.a, m.b, m.missing)

-- 5. a slot the constructor may leave nil, and a name only __index has
local Opt = {fallback = "class value"}
Opt.__index = Opt
local function make(v) return setmetatable({slot = v, fallback = nil}, Opt) end
local x, y = make(nil), make(5)
print(x.slot, y.slot, x.fallback, y.fallback)
y.fallback = "own value"
print(x.fallback, y.fallback)

-- 6. a metatable without __index answers nothing
local plain = setmetatable({v = 1}, {__tostring = function() return "plain" end})
print(plain.v, plain.w, tostring(plain))

-- 7. a class that may hold the field under a key computed at run time
local Dyn = {}
Dyn.__index = Dyn
local d = setmetatable({v = 1}, Dyn)
local key = "sec" .. "ret"
print(d.secret)
Dyn[key] = "hidden"
print(d.secret)

-- 8. indexing a nil receiver names it
local function field_of(a) return a.item end
print(pcall(field_of, nil))
print(pcall(function() local a = nil; if false then a = Tree.new(1, 0) end; return a.item end))
print(pcall(function() local a = nil; if false then a = Tree.new(1, 0) end; return a.left end))
print(pcall(function() local a = nil; if false then a = make(1) end; return a.slot end))

-- 9. the metatable given at birth
local Point = {}
Point.__index = Point
function Point:sum() return self.x + self.y end
local p = setmetatable({x = 1, y = 2}, Point)
print(getmetatable(p) == Point, p:sum())
local q = setmetatable({x = 3}, nil)
print(getmetatable(q), q.x)
print(pcall(setmetatable, {x = 1}, 5))
print(pcall(function() return setmetatable({x = 1}, 5) end))
local guarded = setmetatable({x = 1}, {__metatable = "locked"})
print(getmetatable(guarded), pcall(setmetatable, guarded, {}))
local n = 0
local function count() n = n + 1; return n end
local r = setmetatable({a = count(), b = count()}, (function() n = n * 10; return Point end)())
print(r.a, r.b, n)

-- 10. stores into fields every table is born with
local Acc = {}
Acc.__index = Acc
local acc = setmetatable({total = 0, count = 0}, Acc)
for i = 1, 5 do
  acc.total = acc.total + i
  acc.count = acc.count + 1
end
print(acc.total, acc.count, rawget(acc, "total"))
local keys = {}
for key, value in pairs(acc) do keys[#keys + 1] = key .. "=" .. tostring(value) end
table.sort(keys)
print(table.concat(keys, " "))
