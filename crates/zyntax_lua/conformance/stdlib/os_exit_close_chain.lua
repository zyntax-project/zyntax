-- os.exit(code, true) closes only the main thread's variables, each seeing
-- the error the last one raised, then runs the finalizers.
local x <close> = setmetatable({}, {__close = function (_, e) print("main closed", e) end})
local co = coroutine.wrap(function ()
  local y <close> = setmetatable({}, {__close = function () print("co closed") end})
  local z <close> = setmetatable({}, {__close = function () error("ignored") end})
  os.exit(false, true)
end)
setmetatable({}, {__gc = function () print("finalized") end})
do
  local w <close> = setmetatable({}, {__close = function () print("w closed"); error("boom") end})
  co()
end
print("not reached")
