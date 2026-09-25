-- An object that a call built and worked on, and that nothing holds
-- once the call has returned, is finalized by the next full
-- collection, whether the collection is called at the depth of that
-- call or deeper.
local log = {}
local mt = {__gc = function(o) log[#log + 1] = o.name end}

local function build(name)
  local o = setmetatable({name = name, n = 0}, mt)
  o.n = o.n + 1
  return o.n
end

local function measure(name)
  local o = setmetatable({name = name}, mt)
  return #o.name + #tostring(o.n)
end

local function show()
  table.sort(log)
  print(table.concat(log, " "))
end

build("a")
collectgarbage()
show()

measure("b")
local function deeper() collectgarbage() end
deeper()
show()

for i = 1, 3 do build("c" .. i) end
measure("d")
collectgarbage()
show()
print("finalized", #log)
