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
function A:pass(n)
  for i = 1, n do
    if i == 2 then return nil end
  end
  return self
end
local function probe()
  local t = make()
  return t:pass(3):get()
end
local ok, err = pcall(probe)
print(ok, err:match("nil value") ~= nil)
