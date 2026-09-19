local M = {}
local count = 0
function M.add(a, b) count = count + 1; return a + b end
function M.calls() return count end
M.name = ...
shared_global = "set by mod"
print("loading mod", ...)
return M
