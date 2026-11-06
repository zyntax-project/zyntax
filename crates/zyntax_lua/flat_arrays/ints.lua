local function check(name, got, want)
  if got ~= want then print("FAIL " .. name .. ": got " .. tostring(got) .. " want " .. tostring(want)) end
end
-- build, read, overwrite, length
local xs = {}
for i = 1, 5000 do xs[i] = i * 3 end
check("len", #xs, 5000)
check("read", xs[1] + xs[2500] + xs[5000], 3 + 7500 + 15000)
xs[10] = 7
check("overwrite", xs[10], 7)
check("out of range", xs[5001], nil)
-- append with #
local ys = {}
for i = 1, 100 do ys[#ys + 1] = i end
check("append len", #ys, 100)
-- pop
for i = 1, 40 do ys[#ys] = nil end
check("pop len", #ys, 60)
check("pop read", ys[60], 60)
-- pairs, ipairs, next
local sum, cnt = 0, 0
for k, v in pairs(ys) do sum = sum + k * v; cnt = cnt + 1 end
check("pairs cnt", cnt, 60)
local s2 = 0
for i, v in ipairs(ys) do s2 = s2 + i end
check("ipairs", s2, 60 * 61 // 2)
local k, v = next(ys)
check("next k", k, 1); check("next v", v, 1)
check("next end", next(ys, 60), nil)
-- library
local zs = {}
for i = 1, 10 do zs[i] = 11 - i end
table.sort(zs)
check("sort", zs[1] .. zs[10], "110")
table.insert(zs, 99)
check("insert", zs[11], 99)
table.insert(zs, 1, 0)
check("insert at", zs[1] .. zs[2], "01")
check("remove", table.remove(zs), 99)
check("concat", table.concat({1, 2, 3}, ","), "1,2,3")
local ws = {}
for i = 1, 5 do ws[i] = i end
check("unpack", select("#", table.unpack(ws)), 5)
check("concat flat", table.concat(ws, "-"), "1-2-3-4-5")
-- mixed stores convert
local m = {}
for i = 1, 5 do m[i] = i end
m[6] = "six"
check("mixed str", m[6], "six")
check("mixed keep", m[3], 3)
local f = {}
for i = 1, 5 do f[i] = i end
f[3] = 2.5
check("float conv", f[3], 2.5)
check("float keep", f[4], 4)
local h = {}
for i = 1, 5 do h[i] = i end
h[3] = nil
check("hole", h[3], nil)
check("hole after", h[4], 4)
local g = {}
for i = 1, 5 do g[i] = i end
g.name = "x"
g[6] = 6
check("hash+arr", g.name .. g[6] .. #g, "x66")
-- metatable
local mt = setmetatable({}, {__newindex = function(t, k, v) rawset(t, k, v * 2) end})
for i = 1, 3 do mt[i] = i end
check("newindex", mt[3], 6)
local ix = setmetatable({}, {__index = function(t, k) return -k end})
for i = 1, 3 do rawset(ix, i, i) end
check("index hit", ix[2], 2)
check("index miss", ix[9], -9)
-- negative and zero keys, huge value
local n = {}
n[0] = 1; n[-1] = 2
for i = 1, 3 do n[i] = i end
check("zero key", n[0] + n[-1], 3)
local big = {}
big[1] = math.maxinteger; big[2] = math.mininteger
check("big", big[1] > 0 and big[2] < 0, true)
-- weak table with ints
local weak = setmetatable({}, {__mode = "v"})
for i = 1, 10 do weak[i] = i end
collectgarbage()
check("weak ints", weak[5], 5)
weak[11] = {}
collectgarbage()
-- function taking dynamic table
local function total(t) local s = 0 for i = 1, #t do s = s + t[i] end return s end
check("dyn total", total(xs) > 0, true)
check("dyn total2", total({1, 2, 3}), 6)
-- table.move, rawget, rawlen
local mv = {}
for i = 1, 4 do mv[i] = i end
table.move(mv, 1, 4, 3)
check("move", mv[6], 4)
check("rawget", rawget(ys, 5), 5)
check("rawlen", rawlen(ys), 60)
-- gc stress
for r = 1, 3 do local t = {} for i = 1, 200000 do t[i] = i end collectgarbage() check("gc", t[200000], 200000) end
print("done")
