-- A __tostring that gives no string: the error is named by its type.
local E = setmetatable({}, { __tostring = function(e) return 7 end })
print("start")
error(E)
print("never")
