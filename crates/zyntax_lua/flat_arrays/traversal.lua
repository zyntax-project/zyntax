local function show(t)
  local keys = {}
  for k, v in pairs(t) do keys[#keys + 1] = tostring(k) .. "=" .. tostring(v) end
  table.sort(keys)
  return table.concat(keys, ",") .. " #" .. #t
end
local t = {} for i = 1, 10 do t[i] = i end
for k in pairs(t) do t[k] = nil end
print("clear", show(t))
t = {} for i = 1, 10 do t[i] = i end
for k, v in pairs(t) do if v % 2 == 0 then t[k] = nil end end
print("evens out", show(t))
t = {} for i = 1, 6 do t[i] = i end
t[#t] = nil; t[#t] = nil
print("pop2", show(t))
t[#t + 1] = 100
print("push after pop", show(t))
t[10] = 5
print("sparse", show(t))
t = {} for i = 1, 5 do t[i] = i end
t[7] = 7; t[6] = 6
print("migrate", show(t))
t = {1, 2, 3}
t.x = 1; t[4] = 4
print("ctor+hash", show(t))
t = {} t[1] = 1 t[2] = 2 t[3] = 3
print("seq", show(t), select("#", table.unpack(t)))
t = {} for i = 1, 5 do t[i] = i end
table.insert(t, 3, 99) table.remove(t, 1)
print("ins/rem", show(t))
t = {} for i = 1, 5 do t[i] = -i end
table.sort(t, function(a, b) return a < b end)
print("sort", show(t))
t = {} for i = 1, 5 do t[i] = i end
t[2] = 2.0; t[3] = 3.5
print("float", show(t), math.type(t[2]), math.type(t[1]))
t = {} for i = 1, 5 do t[i] = i end
print(rawequal(t, t), rawlen(t), next({}), type(next(t)))
local u = {} for i = 1, 3 do u[i] = i end
u[1], u[3] = u[3], u[1]
print("swap", show(u))
u = {} u[1] = 10 u[2] = 20
u[1] = nil
print("nil head", show(u))
u = {} for i = 1, 4 do u[i] = i end
print(#u, u[#u], select(2, ipairs(u)) == u)
local c = {} for i = 1, 3 do c[i] = i end
local d = {table.unpack(c)} d[#d + 1] = 4
print("unpack ctor", show(d))
local e = {10, 20, 30}
e[2] = "x"
print("ctor then str", show(e))
local f = {1, 2, 3}
f[#f + 1] = 4.5
print("ctor then float", show(f))
local g = {n = 0}
for i = 1, 4 do g[#g + 1] = i end
print("hash then arr", show(g))
local mtx = setmetatable({}, {__len = function() return 42 end})
for i = 1, 3 do mtx[i] = i end
print("len mm", #mtx, rawlen(mtx))
print(#"abc", ("x"):rep(3), string.format("%d", 7))
