-- Error positions across calls: a raise after another call in the same
-- statement, raises inside callees, error levels, library errors.

local function g(x)
  if x > 100 then error("big") end
  return x
end
local function deep(x)
  error("deep " .. x, 2)
end
local function field_of(t)
  return t.x.y
end

print(pcall(function() local a = g(1) + g(200) end))
print(pcall(function() local a = g(1) + nil end))
print(pcall(function() local a = g(1) .. {} end))
print(pcall(function() local t = {} t.x = g(2) + t.y end))
print(pcall(function() return g(1) + deep(2) end))
print(pcall(function() local r = deep(g(3)) return r end))
print(pcall(function() local v = g(1) + field_of({}) end))
print(pcall(function() local v = g(1) + field_of({ x = {} }) + g(2) end))
print(pcall(function() return string.rep("x", g(1), {}) end))
print(pcall(function() return string.format("%d", g(1) + 0.5) end))
print(pcall(function() return string.rep("ab", g(2)) .. string.format("%q", {}) end))
print(pcall(function()
  local v = g(1) +
    g(2) +
    nil
end))
print(pcall(function()
  local v = g(1)
  local w = v < nil
end))

-- The same raises inside a loop, after calls in earlier iterations.
print(pcall(function()
  local s = 0
  for i = 1, 5 do
    s = s + g(i)
    if i == 4 then s = s .. {} end
  end
end))
print(pcall(function()
  local i = 0
  while g(i) < 3 do
    i = i + 1
  end
  return i + g(500)
end))

-- A method raising after another method call in the statement.
local Obj = {}
Obj.__index = Obj
function Obj.new() return setmetatable({ n = 0 }, Obj) end
function Obj:bump() self.n = self.n + 1 return self end
function Obj:fail() error("fail at " .. self.n) end
function Obj:fail2() error("fail2 at " .. self.n, 2) end
local o = Obj.new()
print(pcall(function() o:bump():bump():fail() end))
print(pcall(function() o:bump():fail2() end))
print(pcall(function() return o:bump().n + o.missing end))

-- Runtime type errors in nested calls.
local function add(a, b) return a + b end
local function twice(a, b) return add(a, b) * 2 end
print(pcall(function() return twice(1, 2) + twice(1, nil) end))
print(pcall(function() return twice(1, 2) + twice("x", {}) end))
