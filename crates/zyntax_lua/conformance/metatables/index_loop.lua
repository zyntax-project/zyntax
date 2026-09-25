-- A chain of table handlers that comes back on itself is refused
-- after a bounded number of steps; a function handler ends a chain.
local function try (f, ...)
  local ok, msg = pcall(f, ...)
  print(ok, msg)
end

-- A table that is its own __index and __newindex.
local a = {}
setmetatable(a, a); a.__index = a; a.__newindex = a
try(function (t, k) return t[k] end, a, 10)
try(function (t, k, v) t[k] = v end, a, 10, true)

-- __newindex only.
local b = setmetatable({}, {})
getmetatable(b).__newindex = b
try(function () b.x = 1 end)
print(rawget(b, "x"))

-- A string key, read and written.
try(function () return a.name end)
try(function () a.name = 1 end)

-- Two tables that name each other.
local c, d = {}, {}
setmetatable(c, {__index = d, __newindex = d})
setmetatable(d, {__index = c, __newindex = c})
try(function () return c[1] end)
try(function () return d.k end)
try(function () c[1] = 2 end)

-- A class whose instances are indexed through a chain that ends.
local Base = {}
Base.__index = Base
function Base.hello () return "hello" end
local Derived = setmetatable({}, Base)
Derived.__index = Derived
local obj = setmetatable({}, Derived)
print(obj.hello(), obj.missing)
obj.field = 3
print(rawget(obj, "field"))

-- A long chain that ends is followed to its end.
local head = {}
local cur = head
for i = 1, 1500 do
  local nxt = {}
  setmetatable(cur, {__index = nxt, __newindex = nxt})
  cur = nxt
end
cur.deep = "found"
print(head.deep)
head.other = 7
print(rawget(cur, "other"))

-- A chain ending in a function handler calls it.
local e = setmetatable({}, {__index = setmetatable({}, {__index = function (_, k) return k .. "!" end})})
print(e.x)

-- The error is positioned at the access.
local ok, msg = pcall(function ()
  local t = a
  return t.x
end)
print(ok, msg)
