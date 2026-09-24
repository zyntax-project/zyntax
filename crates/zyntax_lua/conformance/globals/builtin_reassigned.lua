-- a builtin the program assigns is still the library function before the assignment
local old_print = print
print = function(...) return old_print("wrapped:", ...) end
print("hi", 1)
local old_tostring = tostring
tostring = function(v) return "<" .. old_tostring(v) .. ">" end
print(tostring(5))
local old_setmetatable = setmetatable
local count = 0
setmetatable = function(t, m) count = count + 1; return old_setmetatable(t, m) end
local obj = setmetatable({}, {__index = function() return 7 end})
print(obj.x, count)
local old_require = require
local seen = {}
require = function(name) seen[#seen + 1] = name; return old_require(name) end
print(require("string") == string, seen[1])
local old_type = type
function type(v) return "T:" .. old_type(v) end
print(type(1), old_type(type))
local function show() return tostring(nil) end
print(show())
print = old_print
print("restored")
