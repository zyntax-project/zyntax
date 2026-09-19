-- Tables: array part, hash part, length, insertion, traversal.
local t = { 10, 20, 30 }
print(#t, t[1], t[3], t[4])
t[4] = 40
print(#t, t[4])
t[#t + 1] = 50
print(#t)
t[#t] = nil
print(#t)

local p = { x = 1, y = 2, ["z"] = 3 }
print(p.x, p.y, p.z, p.w)
p.w = 4
print(p.w)
p.x = nil
print(p.x)

local mixed = { 1, 2, a = "A", 3, b = "B" }
print(#mixed, mixed.a, mixed.b, mixed[3])

local sum = 0
for i, v in ipairs(t) do
  sum = sum + v * i
end
print(sum)

local keys = {}
for k, v in pairs(p) do
  keys[#keys + 1] = k .. "=" .. tostring(v)
end
table.sort(keys)
print(table.concat(keys, ","))

local nested = { a = { b = { c = "deep" } } }
print(nested.a.b.c)
nested.a.b.c = "changed"
print(nested["a"]["b"]["c"])

local arr = {}
for i = 1, 10 do arr[i] = i * i end
print(#arr, arr[10])
table.insert(arr, 121)
table.insert(arr, 1, 0)
print(#arr, arr[1], arr[2], arr[12])
print(table.remove(arr), #arr)
print(table.remove(arr, 1), #arr, arr[1])
print(table.concat(arr, " "))
print(table.unpack({ 1, 2, 3 }))
local packed = table.pack(1, nil, 3)
print(packed.n, packed[1], packed[2], packed[3])

local f = { [1.0] = "one", [2] = "two" }
print(f[1], f[2.0], #f)
local s = { [true] = "yes" }
print(s[true])
print(next({}))
print(type(next({ 5 })))
local count = 0
for _ in pairs({ 1, 2, 3, x = 1, y = 2 }) do count = count + 1 end
print(count)
local holes = {}
holes[1] = 1
holes[3] = 3
holes[2] = 2
print(#holes)
print(rawlen({ 1, 2 }), rawget(p, "y"), rawequal(p, p), rawequal(p, {}))
