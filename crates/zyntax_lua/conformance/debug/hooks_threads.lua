-- hooks belong to a thread.
local ev = {}
local function rec(e, l) ev[#ev + 1] = e .. ":" .. tostring(l) end

local co = coroutine.create(function()
  local a = 1
  coroutine.yield(a)
  local b = 2
  return a + b
end)
debug.sethook(co, rec, "l")
print(debug.gethook())
print(debug.gethook(co) == rec, select(2, debug.gethook(co)))
print(coroutine.resume(co))
print(coroutine.resume(co))
print(table.concat(ev, " "))
print(debug.gethook(co) == rec, coroutine.status(co))

-- a thread made under a hook has its mask, not its function
ev = {}
debug.sethook(rec, "l")
local co2 = coroutine.create(function() local x = 1 return x end)
debug.sethook()
print(debug.gethook(co2))
coroutine.resume(co2)
print(#ev > 0, ev[1])

-- a hook set from inside a thread on the main one
local main = coroutine.running()
local co3 = coroutine.create(function()
  debug.sethook(main, rec, "r", 3)
  print("in co3", debug.gethook())
  print("main's", select(2, debug.gethook(main)))
end)
coroutine.resume(co3)
print("main after", select(2, debug.gethook()))
debug.sethook()
print(debug.gethook())

-- a dead thread keeps a hook set on it
local co4 = coroutine.create(function() return 1 end)
coroutine.resume(co4)
print(debug.gethook(co4))
debug.sethook(co4, rec, "c", 3)
print(select(2, debug.gethook(co4)))
print(debug.gethook())

local calls = 0
debug.sethook(function() calls = calls + 1 end, "c")
local function f(n) if n > 0 then return f(n - 1) end return 0 end
f(10)
debug.sethook()
print(calls)
