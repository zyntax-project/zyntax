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
local saved
local C = {__index = A, __newindex = function(t, k, v) saved = t; rawset(t, k, v) end}
local function make()
 local o = A.new()
 setmetatable(o, C)
 o.extra = 7
 return o
end
local probe = make()
setmetatable(saved, B)
print(probe:get())
