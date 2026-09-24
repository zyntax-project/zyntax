-- Two classes whose instances share one shape.
local A = {}; A.__index = A
function A:name() return "A" .. self.n end
function A:bump() self.n = self.n + 1; return self end
local B = {}; B.__index = B
function B:name() return "B" .. self.n end
function B:bump() self.n = self.n + 10; return self end
local function new(cls, n) return setmetatable({n = n}, cls) end
local list = {new(A, 1), new(B, 2), new(A, 3)}
for i = 1, #list do
  print(list[i]:bump():name())
end
local total = 0
for _ = 1, 100 do
  for i = 1, #list do total = total + #list[i]:name() end
end
print(total)
local c = new(A, 0)
setmetatable(c, B)
print(c:bump():name())
