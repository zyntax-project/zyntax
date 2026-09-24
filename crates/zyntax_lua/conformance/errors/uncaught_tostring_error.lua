-- A __tostring that raises: the report is that error's message.
local E = setmetatable({}, { __tostring = function(e) error("inner") end })
print("start")
error(E)
print("never")
