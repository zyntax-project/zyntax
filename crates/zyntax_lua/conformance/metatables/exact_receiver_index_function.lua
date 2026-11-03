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
local C = {__index = function(t, k) return B[k] end}
local function make() return setmetatable({value = 3}, C) end
local probe = make()
print(probe:get())
