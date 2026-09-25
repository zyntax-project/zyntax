-- debug.getmetatable and setmetatable on values that are not tables.
print(debug.getmetatable(1), debug.getmetatable(true), debug.getmetatable(nil))
print(debug.getmetatable("x") == getmetatable("x"))

local numbers = {__index = function(n, k) return k .. "#" .. n end}
print(debug.setmetatable(10, numbers))
print(debug.getmetatable(3) == numbers, getmetatable(2.5) == numbers)
local function get(v, k) return v[k] end
print(get(7, "x"), get(1.5, "y"))
debug.setmetatable(0, nil)
print(debug.getmetatable(1))

local bools = {__index = {name = "a boolean"}}
debug.setmetatable(false, bools)
print(get(true, "name"), get(false, "other"))
debug.setmetatable(true, nil)

debug.setmetatable(nil, {__index = function(_, k) return "nil has " .. k end})
print(get(nil, "y"))
debug.setmetatable(nil, nil)
print(pcall(get, nil, "y"))

local fmeta = {__index = {arity = "unknown"}, __metatable = "locked"}
debug.setmetatable(print, fmeta)
print(get(print, "arity"), getmetatable(print), debug.getmetatable(print) == fmeta)
debug.setmetatable(print, nil)

local t = setmetatable({}, {__metatable = "protected"})
print(pcall(setmetatable, t, {}))
print(debug.setmetatable(t, {mark = 1}) == t, debug.getmetatable(t).mark)
