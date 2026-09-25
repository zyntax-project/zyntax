-- a message handler runs where the error was raised.
local function deep(n)
  if n == 0 then error("x") end
  return 1 + deep(n - 1)
end
print(select(2, xpcall(deep, debug.traceback, 2)))

local function handler(m)
  local levels = 0
  while debug.getinfo(levels + 2, "l") do levels = levels + 1 end
  return m .. " at depth " .. levels .. " line " .. debug.getinfo(2, "l").currentline
end
print(xpcall(deep, handler, 3))

local function lvl()
  local c = 0
  while debug.getinfo(c + 1) do c = c + 1 end
  return c
end
print("levels after", lvl())

local t = setmetatable({}, {__index = function(t, k) error("idx " .. k) end})
print(select(2, xpcall(function() return t.a end, debug.traceback)))

-- a thread an error killed keeps the stack it was raised on
local co = coroutine.create(function(a)
  local x = coroutine.yield(a)
  error("boom")
end)
coroutine.resume(co, 1)
print(debug.traceback(co))
print(coroutine.resume(co))
print(debug.traceback(co))
print(debug.traceback(co, "dead", 1))
print("levels after", lvl())
