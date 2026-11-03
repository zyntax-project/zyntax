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
function A:pass()
  error("method-stop")
  return self
end
function A:after()
  reached = reached + 1
  return self
end
local function probe()
  local t = make()
  return t:pass():after():get()
end
local ok, err = pcall(probe)
print(ok, err:match("method%-stop$") ~= nil, reached)
