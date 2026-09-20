-- coroutine.close runs the <close> handlers a suspended coroutine has pending
local function func2close(f)
  return setmetatable({}, {__close = f})
end
local X
local co = coroutine.create(function ()
  local x <close> = func2close(function (self, err)
    print("closing x", err); X = false
  end)
  X = true
  coroutine.yield()
  print("not reached")
end)
coroutine.resume(co)
print(X, coroutine.status(co))
print(coroutine.close(co))
print(X, coroutine.status(co))
-- nested blocks close innermost first, through pcall and function calls
local log = {}
co = coroutine.create(function ()
  local a <close> = func2close(function () log[#log + 1] = "a" end)
  do
    local b <close> = func2close(function () log[#log + 1] = "b" end)
    pcall(function ()
      local c <close> = func2close(function () log[#log + 1] = "c" end)
      coroutine.yield(1)
      log[#log + 1] = "after yield"
    end)
    log[#log + 1] = "after pcall"
  end
end)
print(coroutine.resume(co))
print(coroutine.close(co))
print(table.concat(log, " "), coroutine.status(co))
-- an error in a handler is the result of close
co = coroutine.create(function ()
  local x <close> = func2close(function () error("in close") end)
  coroutine.yield()
end)
coroutine.resume(co)
local st, msg = coroutine.close(co)
print(st, msg)
print(coroutine.close(co))
-- a coroutine cannot close itself while closing
co = coroutine.create(function ()
  local x <close> = func2close(function ()
    print(pcall(coroutine.close, co))
  end)
  coroutine.yield(20)
end)
print(coroutine.resume(co))
print(coroutine.close(co))
-- closing after an error reports it once
co = coroutine.create(error)
print(coroutine.resume(co, 100))
print(coroutine.close(co))
print(coroutine.close(co))
-- never started, or already finished: nothing to do
co = coroutine.create(function () local x <close> = func2close(print) end)
print(coroutine.close(co), coroutine.status(co))
co = coroutine.create(function () return 1 end)
coroutine.resume(co)
print(coroutine.close(co))
-- the value yielded is not taken as a resume
co = coroutine.wrap(function () local n = 0; while true do n = n + coroutine.yield(n) end end)
print(co(), co(2), co(3))
