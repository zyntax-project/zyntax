-- the variable a type error names
print(pcall(function() local x; return x.field end))
print(pcall(function() return undefined_global.field end))
local up
print(pcall(function() return up.field end))
local t = {}
print(pcall(function() return t.a.b end))
print(pcall(function() return t.a + 1 end))
print(pcall(function() local s = "x"; return s .. t.a end))
print(pcall(function() return #t.a end))
print(pcall(function() return ipairs() end))
