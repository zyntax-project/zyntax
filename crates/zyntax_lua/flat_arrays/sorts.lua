local function dump(t) local o = {} for i = 1, #t do o[#o + 1] = tostring(t[i]) end return table.concat(o, " ") end
local seed = 12345
local function rnd(n) seed = (seed * 1103515245 + 12345) % 2147483648 return seed % n end
local t = {} for i = 1, 40 do t[i] = rnd(10) end
table.sort(t) print("ints dup", dump(t))
t = {} for i = 1, 30 do t[i] = 31 - i end
table.sort(t) print("reversed", dump(t))
t = {} for i = 1, 30 do t[i] = rnd(1000) / 8 end
table.sort(t) print("floats", dump(t))
t = {0.0, -0.0, 1.5, -0.0, 0.0, -2.5}
table.sort(t) print("zeros", dump(t))
t = {3.5, 1/0, -1/0, 2.5}
table.sort(t) print("inf", dump(t))
local ok, err = pcall(function() local n = {1.5, 0/0, 3.5, 0/0, 2.5, 0/0, 9.5, 8.5} table.sort(n) return dump(n) end)
print("nan", ok, ok and err or "err")
t = {} for i = 1, 5 do t[i] = i end table.sort(t, function(a, b) return a > b end) print("desc", dump(t))
t = {5} table.sort(t) print("one", dump(t))
t = {} table.sort(t) print("empty", #t)
t = {} for i = 1, 200000 do t[i] = rnd(1000000) end
table.sort(t) local okk = true for i = 2, #t do if t[i - 1] > t[i] then okk = false end end
print("big", okk, #t)
t = {} for i = 1, 100000 do t[i] = rnd(1000000) + 0.5 end
table.sort(t) okk = true for i = 2, #t do if t[i - 1] > t[i] then okk = false end end
print("bigf", okk)
t = {1, 2, 3} t[2] = "x"
print("mixed err", pcall(table.sort, t))
