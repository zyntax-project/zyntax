-- the main chunk is a vararg function of the script arguments
print(select("#", ...), ...)
local a = ...
print(a, #{...}, type(arg), #arg)
local function count(...) return select("#", ...) end
print(count(...), count(..., 1), count(1, ...))
