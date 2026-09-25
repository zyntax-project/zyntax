-- errors raised in C and errors passing through C
package.cpath = "./?.so"
local m = require "cerrors"

print(pcall(m.boom))
print(pcall(function() m.boom(7) end))
local ok, e = pcall(function()
  local x = 1
  m.boom(x + 1)
end)
print(ok, e)
local ok, v = pcall(m.throw, { code = 3 })
print(ok, type(v), v.code)
print(pcall(m.throw, nil))
print(pcall(m.throw, 12.5))

-- C calling Lua that errors
print(pcall(m.call, function(a, b) return a + b end, 2, 3))
print(pcall(m.call, function() error("from lua") end))
ok, v = pcall(m.call, function() error({ 1 }) end)
print(ok, type(v), v[1])
print(pcall(m.call, function() local t = nil; return t.x end))
print(m.pcall(function() error("caught in c") end))
print(m.pcall(function(...) return ... end, 1, 2, 3))
print(m.pcall(error, "plain", 0))
print(m.xpcall(function() error("with handler", 0) end))
print(m.xpcall(function() return "fine" end))

-- C to Lua to C
print(pcall(m.call, m.boom))
print(pcall(m.call, function() return m.call(m.boom, 5) end))
print(m.pcall(m.call, m.throw, "deep"))
print(m.outer())
print(pcall(m.crowded))
print(m.call(function() return "stack is clean" end))

print(pcall(m.checkint, 21))
print((pcall(m.checkint, "x")))

-- in a coroutine
local co = coroutine.wrap(function(x)
  local ok, e = pcall(m.boom, x)
  coroutine.yield(e)
  return m.call(function(a) return a * 2 end, x)
end)
print(co(5))
print(co())
print(pcall(coroutine.wrap(function() m.boom(9) end)))
