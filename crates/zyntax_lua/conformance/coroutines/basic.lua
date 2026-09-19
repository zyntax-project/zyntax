-- Coroutines: create, resume, yield, status, wrap, values both ways.
local co = coroutine.create(function(a, b)
  print("start", a, b)
  local c = coroutine.yield(a + b)
  print("got", c)
  local d, e = coroutine.yield(c * 2)
  print("got", d, e)
  return "done", a + b + c
end)
print(coroutine.status(co))
print(coroutine.resume(co, 1, 2))
print(coroutine.status(co))
print(coroutine.resume(co, 10))
print(coroutine.resume(co, "x", "y"))
print(coroutine.status(co))
print(coroutine.resume(co))

local function range(n)
  return coroutine.wrap(function()
    for i = 1, n do coroutine.yield(i) end
  end)
end
local sum = 0
for v in range(10) do sum = sum + v end
print(sum)

local gen = coroutine.wrap(function()
  local a, b = 0, 1
  while true do
    coroutine.yield(a)
    a, b = b, a + b
  end
end)
local fibs = {}
for i = 1, 12 do fibs[i] = gen() end
print(table.concat(fibs, " "))

-- Yield from a nested call inside the coroutine.
local function helper(x)
  coroutine.yield(x * 10)
  return x + 1
end
local nested = coroutine.create(function()
  local r = helper(1)
  local s = helper(r)
  return s
end)
print(coroutine.resume(nested))
print(coroutine.resume(nested))
print(coroutine.resume(nested))
print(coroutine.status(nested))

print(coroutine.isyieldable())
local inner = coroutine.create(function()
  print(coroutine.isyieldable())
  print(coroutine.running() ~= nil)
  local outer_status = coroutine.status(co)
  print(outer_status)
end)
coroutine.resume(inner)
print(coroutine.status(inner))

-- A coroutine yielding several values and receiving several.
local multi = coroutine.wrap(function(...)
  local args = { ... }
  while true do
    local n = select("#", coroutine.yield(#args, args[1]))
    args = { n }
  end
end)
print(multi("a", "b", "c"))
print(multi(1, 2))
print(multi())

-- Producers and consumers.
local function producer()
  return coroutine.create(function()
    for _, item in ipairs({ "x", "y", "z" }) do
      coroutine.yield(item)
    end
    return nil
  end)
end
local p = producer()
while true do
  local ok, item = coroutine.resume(p)
  if not item then break end
  io.write(item, " ")
end
print()
print(coroutine.status(p))
local dead = coroutine.create(function() end)
coroutine.resume(dead)
print(coroutine.resume(dead))
print(coroutine.close(dead))
