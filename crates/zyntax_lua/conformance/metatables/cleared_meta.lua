-- An instance whose metatable is taken away no longer finds methods.
local C = {}
C.__index = C
function C.new(v) return setmetatable({v = v}, C) end
function C:get() return self.v end
local a, b = C.new(1), C.new(2)
print(a:get(), b:get())
setmetatable(b, nil)
print(a:get(), pcall(function() return b:get() end))
