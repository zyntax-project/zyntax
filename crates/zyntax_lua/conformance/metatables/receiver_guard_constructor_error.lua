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
local reached = 0
local function bad()
  return setmetatable({value = error("constructor-stop")}, A)
end
function A:after()
  reached = reached + 1
  return self
end
local function probe()
  local t = bad()
  return t:after():get()
end
local ok, err = pcall(probe)
print(ok, err:match("constructor%-stop$") ~= nil, reached)
