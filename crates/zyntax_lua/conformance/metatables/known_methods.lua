-- A method stored before a call is found where the call stands; one
-- stored later, in one branch, or overriding after a call is still
-- looked up.
local Base = {}
Base.__index = Base
function Base.new(v) return setmetatable({v = v}, Base) end
function Base:get() return self.v end
local k = 7
function Base:plus() return self.v + k end

local Sub = setmetatable({}, {__index = Base})
Sub.__index = Sub
function Sub.new(v)
  local o = Base.new(v)
  setmetatable(o, Sub)
  o.extra = 1
  return o
end

local b, s = Base.new(1), Sub.new(2)
print(b:get(), s:get(), b:plus(), s:plus())
print(pcall(function() return s:twice() end))
function Sub:twice() return self.v * 2 end
print(s:twice())
print(pcall(function() return b:twice() end))
function Sub:get() return -self.v end
print(b:get(), s:get())
if k > 100 then
  function Base:maybe() return 1 end
end
print(pcall(function() return b:maybe() end))
local total = 0
for i = 1, 100 do
  total = total + b:get() + s:get() + s:twice() + b:plus() + i
end
print(total)
