-- # on a table is whatever __len returns, called with the operand twice
local cap
local function f(op)
  return function (...) cap = {[0] = op, ...}; return (...) end
end
local mt = {__len = f("len")}
local b = setmetatable({}, mt)
print(#b == b, cap[0], cap[1] == b, cap[2] == b, #cap)
local s = setmetatable({}, {__len = function () return "seven" end})
print(#s, type(#s))
local n = setmetatable({1, 2, 3}, {__len = function (t) return 2.5 end})
print(#n, math.type(#n))
local raw = setmetatable({1, 2, 3}, {__len = function (t) return rawlen(t) * 2 end})
print(#raw, rawlen(raw))
-- a plain table alongside keeps its border, and the loop idioms work
local t = {10, 20, 30}
print(#t)
for i = 1, #t do io.write(t[i], " ") end print()
t[#t + 1] = 40
print(#t, t[#t])
while #t > 0 do t[#t] = nil end
print(#t)
-- the table library refuses a non-integer length
print(pcall(table.insert, s, 1))
print(pcall(table.insert, n, 1))
-- __len on a string's metatable
getmetatable("").__len = function (s) return "len of " .. s end
print(#"abc")
getmetatable("").__len = nil
print(#"abc")
