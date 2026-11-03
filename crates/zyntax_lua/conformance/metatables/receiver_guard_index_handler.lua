local A = {}
A.__index = A
local B = {}
B.__index = B
local function make()
  return setmetatable({value = 1}, A)
end
local function other()
  return setmetatable({value = 2}, B)
end
function A:get() return self.value + 10 end
function B:get() return self.value + 20 end
local function pass(self) return other() end
local C = {__index = function(t, key) return pass end}
local function construct() return setmetatable({value = 1}, C) end
local function probe()
  local t = construct()
  return t:pass():get()
end
print(probe())
