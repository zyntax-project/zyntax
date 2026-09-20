-- a function declared into a table is called through it: its parameters take anything
local M = {}
function M.check(x) if x then return "yes" else return "no" end end
function M.eq(a, b) return a == b end
function M:twice(x) return x * 2 end
print(M.check(1), M.check(nil), M.check(false))
print(M.eq(1, 1), M.eq(1, 2), M.eq("a", "a"), M.eq(nil, nil))
print(M:twice(4), M.twice(nil, 2.5))
local o = { n = 3 }
function o.sum(a, b, c) return a + b + c end
print(o.sum(1, 2, 3), o.sum(1.5, 2, 3))
