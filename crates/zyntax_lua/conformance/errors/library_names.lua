local function e(...) print(pcall(...)) end
e(string.format, "%q", {})
e(string.format, "%10s", "\0")
e(string.format, "%s", "\0")
local m = setmetatable({}, {__tostring = function() return {} end})
e(tostring, m)
e(table.concat, 3)
e(table.concat, {}, " ", math.maxinteger, math.maxinteger)
e(table.concat, {}, " ", math.mininteger, math.mininteger)
