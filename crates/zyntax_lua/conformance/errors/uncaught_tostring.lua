-- An uncaught table error with __tostring reports what it gives.
local E = setmetatable({}, { __tostring = function(e) return "custom error" end })
print("start")
error(E)
print("never")
