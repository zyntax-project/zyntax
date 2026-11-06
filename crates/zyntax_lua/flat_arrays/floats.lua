local function show(t)
  local keys = {}
  for k, v in pairs(t) do keys[#keys + 1] = tostring(k) .. "=" .. tostring(v) .. ":" .. (math.type(v) or type(v)) end
  table.sort(keys)
  return table.concat(keys, ",") .. " #" .. #t
end
local t = {}
for i = 1, 6 do t[i] = i / 4 end
print("build", show(t))
t[2] = 9.5; print("overwrite", show(t))
t[3] = 7; print("int into floats", show(t))
t[#t + 1] = 0.125; print("append", show(t))
local u = {}
for i = 1, 5 do u[#u + 1] = i * 1.5 end
for i = 1, 3 do u[#u] = nil end
print("pop", show(u))
local c = {1.5, 2.5, 3.5}
print("ctor", show(c), c[2] + c[3])
c[2] = "s"; print("ctor str", show(c))
local d = {0.0, -0.0, 1/0, -1/0}
print("special", d[1], d[2], d[3], d[4], 1/d[2])
local nan = {0/0}
print("nan", nan[1] ~= nan[1])
local e = {}
for i = 1, 5 do e[i] = (6 - i) * 0.5 end
table.sort(e); print("sort", show(e))
table.insert(e, 1, 100.5); print("insert", show(e))
print("remove", table.remove(e), show(e))
print("concat", table.concat({1.5, 2.5}, " "), select("#", table.unpack({1.5, 2.5, 3.5})))
local function sum(x) local s = 0 for i = 1, #x do s = s + x[i] end return s end
print("dyn sum", sum({0.5, 1.5}), sum(e))
local m = {}
for i = 1, 4 do m[i] = i * 0.5 end
for k, v in pairs(m) do m[k] = v * 2 end
print("pairs write", show(m))
for k, v in ipairs(m) do m[k] = nil end
print("ipairs clear", show(m))
local g = {}
for i = 1, 4 do g[i] = i + 0.5 end
g.name = "g"; g[5] = 5.5
print("hash+arr", show(g))
local w = setmetatable({}, {__mode = "v"})
for i = 1, 5 do w[i] = i + 0.25 end
collectgarbage()
print("weak", show(w))
local mt = setmetatable({}, {__newindex = function(t, k, v) rawset(t, k, v + 1) end})
for i = 1, 3 do mt[i] = i + 0.5 end
print("newindex", show(mt))
local big = {}
for i = 1, 100000 do big[i] = i + 0.5 end
collectgarbage()
local acc = 0
for i = 1, #big do acc = acc + big[i] end
print("big", acc)
local mixed = {}
for i = 1, 4 do mixed[i] = i end
mixed[2] = 2.5
print("int then float", show(mixed))
local fl = {}
for i = 1, 4 do fl[i] = i + 0.5 end
fl[2] = 2
print("float then int", show(fl), math.type(fl[2]), math.type(fl[3]))
print(math.type(({1.0})[1]), math.type(({1})[1]))
