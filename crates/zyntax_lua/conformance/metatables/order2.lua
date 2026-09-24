-- The same with plain methods and with a method two classes up.
local A = {}; A.__index = A
function A:m() return "m1" end
local B = setmetatable({}, A); B.__index = B
local o = setmetatable({}, B)
local function swap()
  A.m = function() return "m2" end
  return true
end
print(swap and o:m(swap()), o:m())
function A:m() return "m1" end
local function shadow()
  B.m = function() return "b" end
  return 1
end
print(o:m(shadow()), o:m())
B.m = nil
local function reset()
  A.m = nil
  return 1
end
print(o:m(reset()))
print(pcall(o.m, o))
