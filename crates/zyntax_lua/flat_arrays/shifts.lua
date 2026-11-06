local function dump(t) local o = {} for i = 1, #t do o[#o + 1] = tostring(t[i]) .. (math.type(t[i]) == "float" and "f" or "") end return table.concat(o, " ") .. " #" .. #t end
local t = {} for i = 1, 6 do t[i] = i * 10 end
table.insert(t, 1, 5) print("ins head", dump(t))
table.insert(t, 4, 35) print("ins mid", dump(t))
table.insert(t, #t + 1, 99) print("ins end", dump(t))
table.insert(t, 100) print("append", dump(t))
table.insert(t, 2, 1.5) print("ins float into ints", dump(t))
table.insert(t, 2, "s") print("ins str", dump(t))
print("rm mid", table.remove(t, 3), dump(t))
print("rm head", table.remove(t, 1), dump(t))
print("rm last", table.remove(t), dump(t))
local f = {} for i = 1, 5 do f[i] = i + 0.5 end
table.insert(f, 2, 9.25) print("f ins", dump(f))
table.insert(f, 2, 3) print("f ins int", dump(f))
local g = {} for i = 1, 5 do g[i] = i + 0.5 end
print("f rm", table.remove(g, 2), table.remove(g), dump(g))
local s = {} for i = 1, 5 do s[i] = i end
while #s > 0 do io.write(table.remove(s), " ") end print()
local e = {} for i = 1, 3 do e[i] = i end
for i = 1, 3 do table.remove(e, 1) end print("emptied", dump(e), table.remove(e), table.remove(e, 0))
local h = {} for i = 1, 4 do h[i] = i end h.k = 1
table.insert(h, 2, 7) print("hash ins", dump(h), h.k)
local ok, err = pcall(table.insert, {1, 2, 3}, 9, 1) print("oob", ok, err)
ok, err = pcall(table.remove, {1, 2, 3}, 9) print("oob rm", ok, err)
local m = setmetatable({}, {__index = function(_, k) return nil end})
for i = 1, 4 do m[i] = i end
table.insert(m, 2, 50) print("meta ins", dump(m))
print("rm meta", table.remove(m, 1), dump(m))
local q = {} for i = 1, 100000 do q[i] = i end
for i = 1, 1000 do table.insert(q, 1, -i) end
for i = 1, 500 do table.remove(q, 1) end
print("big", #q, q[1], q[#q])
