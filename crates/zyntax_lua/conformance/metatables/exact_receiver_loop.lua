local A = {}
A.__index = A
function A.new()
  return setmetatable({value = 1}, A)
end
function A:get() return self.value end
local B = setmetatable({}, {__index = A})
B.__index = B
function B.new()
  local o = A.new()
  setmetatable(o, B)
  o.value = 2
  return o
end
function B:get() return self.value + 10 end
local function make(n)
 local o = A.new()
 for i = 1, n do setmetatable(o, B) end
 return o
end
local probe = make(0)
print(probe:get())
