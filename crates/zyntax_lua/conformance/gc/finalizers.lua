-- __gc runs for unreachable objects, the last marked first; the
-- object is resurrected for its finalizer, and a __gc added to the
-- metatable after setmetatable does not mark the object. The order
-- depends on when a collection runs (reference Lua collecting after the
-- second object prints 2 1 3), so only the program's own collections run.
collectgarbage("stop")
local order = {}
local saved
local function make()
  for i = 1, 3 do
    setmetatable({i = i}, {__gc = function(o) order[#order + 1] = o.i end})
  end
  local late = setmetatable({}, {})
  getmetatable(late).__gc = function() order[#order + 1] = "late" end
  setmetatable({name = "phoenix"}, {__gc = function(o) saved = o end})
end
make()
collectgarbage()
print(table.concat(order, " "))
print(saved and saved.name)
saved = nil
collectgarbage()
print("finalized once", #order)
local x = setmetatable({}, {__gc = function() print("at exit") end})
local y = setmetatable({}, {__gc = function() print("at exit, marked last") end})
print("end of chunk")
