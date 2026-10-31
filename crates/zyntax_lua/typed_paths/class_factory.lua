-- A class table a function makes: no store of it in the main chunk
-- settles the lookup, so the call dispatches on the receiver's shape.
local function class()
  local C = {}
  C.__index = C
  function C.new(x) return setmetatable({x = x}, C) end
  function C:get() return self.x end
  return C
end
local A = class()
local total = 0
for i = 1, 3 do total = total + A.new(i):get() end
print(total)
