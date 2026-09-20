-- 1. method called before its definition
local T = {}; T.__index = T
local o = setmetatable({v = 1}, T)
print(pcall(function() return o:m() end))
function T:m() return self.v end
print(o:m())
-- 2. own-field override
o.m = function(self) return "own" end
print(o:m())
o.m = nil
print(o:m())
-- 3. two-level inheritance with an override in the middle
local A = {}; A.__index = A
function A:who() return "A" end
function A:hello() return "hello from " .. self:who() end
local B = setmetatable({}, {__index = A}); B.__index = B
function B:who() return "B" end
local C = setmetatable({}, {__index = B}); C.__index = C
local a, b, c = setmetatable({}, A), setmetatable({}, B), setmetatable({}, C)
print(a:hello(), b:hello(), c:hello())
-- 4. __index another table
local Methods = {}
function Methods:get() return self.x * 2 end
local mt = {__index = Methods}
local p = setmetatable({x = 21}, mt)
print(p:get())
-- 5. the method defined while the arguments run
local D = {}; D.__index = D
local d = setmetatable({}, D)
local function define() function D:late(y) return y end return 5 end
print(pcall(function() return d:late(define()) end))
print(d:late(6))
-- 6. metatable replaced by a table from the library side
local E = {}; E.__index = E
function E:f() return "E" end
local e = setmetatable({}, E)
print(e:f())
setmetatable(e, {__index = function(t, k) return function() return "dyn " .. k end end})
print(e:f())
-- 7. method removed
local F = {}; F.__index = F
function F:g() return "g" end
local f = setmetatable({}, F)
print(f:g())
rawset(F, "g", nil)
print(pcall(function() return f:g() end))
-- 8. __index function on the class
local G = setmetatable({}, {__index = function(t, k) return function() return "G." .. k end end})
G.__index = G
local g = setmetatable({}, G)
print(g:anything())
-- 9. methods with upvalues made in a function
local function make(n)
  local K = {}; K.__index = K
  function K:n() return n end
  function K:twice() return self:n() * 2 end
  return setmetatable({}, K)
end
print(make(7):twice(), make(8):n())
-- 10. inherited method through a class made in a loop
local sum = 0
for i = 1, 3 do
  local Cls = setmetatable({}, {__index = A}); Cls.__index = Cls
  local inst = setmetatable({}, Cls)
  sum = sum + #inst:hello()
end
print(sum)
