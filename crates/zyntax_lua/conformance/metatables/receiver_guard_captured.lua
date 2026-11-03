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
function A:pass()
  local function replace() self = other() end
  replace()
  return self
end
local function probe()
  local t = make()
  return t:pass():get()
end
print(probe())
