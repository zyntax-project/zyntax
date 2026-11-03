-- A method inherited through a metatable chain, called on tables of two
-- shapes. The constructors set different fields, and `B.new` may resolve
-- to `A.new` when looked up, so the call's result is not one shape.
local A = {}
A.__index = A
function A.new(s) return setmetatable({ state = s }, A) end
function A:value() return self.state end
function A:flip() self.state = not self.state; return self end

local B = setmetatable({}, { __index = A })
B.__index = B
function B.new(s, m) return setmetatable({ state = s, count = 0, max = m }, B) end
function B:flip()
  self.count = self.count + 1
  if self.count >= self.max then
    self.state = not self.state
    self.count = 0
  end
  return self
end

local a = A.new(true)
local b = B.new(true, 2)
print(a:value(), b:value())
print(a:flip():value(), b:flip():value(), b:flip():value())
local v = true
for _ = 1, 5 do v = a:flip():value() end
print(v, b.count, b:value())
