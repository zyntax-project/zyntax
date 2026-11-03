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
local function make(x)
 if x then return setmetatable({value = 1}, A) else return B.new() end
end
local probe = make(false)
print(probe:get())
