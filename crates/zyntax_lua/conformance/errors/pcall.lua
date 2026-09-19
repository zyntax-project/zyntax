-- error values through pcall
print(pcall(error))
print(pcall(error, nil))
print(pcall(error, "plain", 0))
print(pcall(error, 42))
print(pcall(error, true))
local t = {}
local ok, e = pcall(error, t)
print(ok, e == t)
print(pcall(function() return 1, 2, 3 end))
print(pcall(function() end))

-- positions: level 1 is the caller of error, level 2 its caller
local function f() error("in f") end
print(pcall(f))
local function g() error("in g", 2) end
local function h() g() end
print(pcall(h))
print(pcall(function() error("no position", 0) end))

-- nested pcall: the inner one catches, the outer sees a result
print(pcall(function()
  local ok, e = pcall(error, "inner")
  return ok, e, "after"
end))

-- rethrow
print(pcall(function()
  local ok, e = pcall(error, "first")
  error(e, 0)
end))

-- runtime errors have positions and Lua's wording
print(pcall(function() return 1 + nil end))
print(pcall(function() return #5 end))
print(pcall(function() local s = "a"; return s < 1 end))
print(pcall(function() return ("x"):rep("a") end))
print(pcall(function() return string.rep("x", "a") end))
print(pcall(function() return math.floor({}) end))
print(pcall(function() return tonumber("10", 99) end))

-- assert
print(pcall(assert, false))
print(pcall(assert, nil, "custom"))
print(pcall(assert, false, 7))
print(pcall(assert, 1, 2, 3))
print(select("#", pcall(assert, true)))

-- xpcall: the handler sees the value, its result is returned
print(xpcall(function() error("boom") end, function(m) return "handled: " .. m end))
print(xpcall(function() return "fine", 2 end, print))
print(xpcall(error, function(m) return type(m) end, {}))
print(xpcall(function(a, b) return a + b end, print, 3, 4))

-- errors in coroutines
local co = coroutine.create(function() error("in co") end)
print(coroutine.resume(co))
print(coroutine.status(co))
local w = coroutine.wrap(function() error("in wrap", 0) end)
print(pcall(w))

-- an error stops the statement it is in
local n = 0
print(pcall(function()
  n = n + 1
  error("stop")
  n = n + 1
end))
print(n)

-- an error object is returned as is
local obj = setmetatable({}, { __tostring = function() return "OBJ" end })
local ok2, e2 = pcall(error, obj)
print(ok2, tostring(e2), e2 == obj)

-- pcall of a non-function
print(pcall(5))
print(pcall(nil))
