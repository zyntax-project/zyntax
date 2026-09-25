-- metatables debug.setmetatable gives booleans and nil.
debug.setmetatable(true, {
  __len = function() return 42 end,
  __tostring = function(b) return b and "T" or "F" end,
  __index = function(b, k) return k .. "!" end,
  __concat = function(a, b) return "cat" end,
})
local T, F = true, false
print(#T, #F)
print(tostring(true), tostring(false))
print(T.x)
print(T .. "s")
print(select("#", tostring(true)))
debug.setmetatable(true, nil)
print(tostring(true), pcall(function() return #T end))
debug.setmetatable(nil, {__index = function(_, k) return "nil has " .. k end})
local n = nil
print(n.y)
debug.setmetatable(nil, nil)
print(pcall(function() return n.y end))
