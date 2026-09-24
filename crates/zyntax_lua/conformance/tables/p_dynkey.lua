-- a store under a key not known at compile time may name a field, a
-- name that is none, or a key of another kind

local u = { x = 1, y = 2 }
local function key(c) if c then return "x" end return "s" end
local k = key(false)
u[k] = "s"
print(u[k])
k = key(true)
u[k] = 2.5
print(u.x + 1, u.y)

local held = {}
local flag, obj = true, {}
held[flag] = 1
held[obj] = { w = 2 }
held[obj].w = held[obj].w + held[flag]
print(held[true], held[obj].w, held[false])

local byname = {}
for i = 1, 3 do byname["k" .. i] = i * 10 end
local sum = 0
for i = 1, 4 do sum = sum + (byname["k" .. i] or 0) end
print(sum, byname.k2, byname.k4)
