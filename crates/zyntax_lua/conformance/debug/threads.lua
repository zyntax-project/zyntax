-- The debug library over coroutines: a suspended coroutine's stack.
local co = coroutine.create(function(a)
  local function inner()
    coroutine.yield(a)
  end
  inner()
  return "done"
end)
print(coroutine.resume(co, 1))
print(debug.traceback(co))
print(debug.traceback(co, "with a message"))
print(debug.traceback(co, "from level 1", 1))
local info = debug.getinfo(co, 1, "Sl")
print(info.currentline, info.linedefined, info.what)
print(debug.getinfo(co, 0, "S").what)
print(coroutine.resume(co))
print(debug.traceback(co))

local wrapped = coroutine.wrap(function()
  print(debug.traceback("inside"))
end)
wrapped()
