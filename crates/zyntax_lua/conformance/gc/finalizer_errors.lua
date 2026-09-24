-- An error in a finalizer is a warning, never an error of the program,
-- and every collectgarbage option fails inside a finalizer.
warn("@on")
local ran = {}
local function make()
  setmetatable({}, {__gc = function() ran[#ran + 1] = "first"; error("boom") end})
  setmetatable({}, {__gc = function() ran[#ran + 1] = "second"; error({}) end})
  setmetatable({}, {__gc = function() ran[#ran + 1] = "third" end})
end
make()
collectgarbage()
table.sort(ran)
print(table.concat(ran, " "))
print(pcall(error, "still fine"))
-- Finalized when the program ends.
Keep = setmetatable({}, {__gc = function()
  print("inside", collectgarbage("count"), collectgarbage("step"), collectgarbage())
end})
print("done")
