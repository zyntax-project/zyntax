-- calls from the library into the program nest on a C stack of bounded depth
local depth = 0
local function loop()
  depth = depth + 1
  assert(pcall(loop))
end
local ok, msg = pcall(loop)
print(ok, depth > 100, depth < 300, msg:sub(-16))
print(select(2, msg:gsub("c_stack.lua:5: ", "")) > 100)

-- metamethods count as well
local n = 0
local function meta()
  n = n + 1
  return setmetatable({}, {__index = function () return meta() end}).x
end
print(pcall(meta))
print(n > 100 and n < 300)

-- and resuming a coroutine; wrap adds its caller's position to a string error
local function co() n = n + 1; return coroutine.wrap(co)() end
n = 0
local ok2, msg2 = pcall(co)
print(ok2, msg2:sub(-16), n > 50 and n < 300)

-- a handler that raises is called on its own error; a stack overflow in
-- the handler is an error in error handling
print(xpcall(error, function (m) error("x") end, "boom"))
local k = 0
print(xpcall(error, function (m) k = k + 1; if k < 3 then error("again") end return "h:" .. tostring(m) end, "boom"))
print(k)
local function deep() return 1 + deep() end
print(xpcall(deep, deep))
print(xpcall(deep, function (m) return "handled: " .. m:sub(-14) end))

-- assert positions its message like error; through pcall it has no caller
print(pcall(function () assert(false, "boom") end))
print(pcall(assert, false, "boom"))
print(type(select(2, pcall(function () assert(nil, {}) end))))
print(pcall(function () assert(false, 42) end))

-- wrap: the caller's position in front of a string error, not another value
local f = coroutine.wrap(function () error("x") end)
print(pcall(f))
local g = coroutine.wrap(function () error("y") end)
print(pcall(function () return g() end))
local h = coroutine.wrap(function () error({}) end)
print(select("#", pcall(function () return h() end)))

-- a chain of __call handlers
local function target(...)
  local kinds = {}
  for i = 1, select("#", ...) do kinds[i] = type((select(i, ...))) end
  return table.concat(kinds, " ")
end
local c = target
for i = 1, 5 do c = setmetatable({i}, {__call = c}) end
print(c("a", "b"))
local bad = setmetatable({}, {__call = setmetatable({}, {})})
print(pcall(bad))
print(pcall(setmetatable({}, {__call = 5})))
