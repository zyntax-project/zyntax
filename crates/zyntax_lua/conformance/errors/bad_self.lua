-- A library function reached by a method call names its first
-- argument `self` and counts the rest from it.
local function try(f, ...)
  local ok, err = pcall(f, ...)
  print(ok, err)
end

local aaa = setmetatable({}, {__index = string})
try(function () return aaa:sub() end)
try(function () return aaa:rep(3) end)
try(function () return aaa:byte() end)
local sss = setmetatable({}, {__index = {sub = string.sub, upper = string.upper}})
try(function () return sss:upper() end)
try(function () return sss:sub(1, 2) end)
local s = setmetatable({}, {__index = function (_, k) return string[k] end})
try(function () return s:len() end)

-- The receiver is fine: later arguments count from one.
local function later(x) return x:sub({}) end
try(later, "hello")
try(function () return ("x"):sub({}) end)
local box = {text = "hello"}
try(function () return box.text:rep({}) end)

-- Not a method call: no self, and the arguments count from the first.
local function says(ok, err)
  print(ok, string.find(err, "bad self") ~= nil, string.match(err, "#%d"))
end
local f = string.sub
says(pcall(function () return f({}) end))
says(pcall(function () return string.sub("a", {}) end))

-- A method call whose function calls back into the program leaves
-- no mark behind for the calls it makes.
local g = string.rep
local obj = setmetatable({}, {__index = {each = function (_, h) return h() end}})
says(pcall(function () return obj:each(function () return g("x", {}) end) end))
local list = setmetatable({3, 1, 2}, {__index = table})
says(pcall(function () return list:sort(function (a, b) return g("x", {}) end) end))
says(pcall(function () return g("x", {}) end))
local fresh = setmetatable({3, 1, 2}, {__index = table})
fresh:sort()
print(fresh:concat(","))
