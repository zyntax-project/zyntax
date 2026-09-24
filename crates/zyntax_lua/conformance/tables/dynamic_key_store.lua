-- a store under a string key not known at compile time may name any
-- field, or a name that is none

local t = { x = 1, y = 2, name = "p" }
print(t.x + t.y)
local k = "y"
t[k] = 2.5
print(t.x + t.y)
local j = "z"
t[j] = 7
local z = t.z
if z then print(z + 1) end
local n = "name"
t[n] = nil
print(t.name, t.x)

local fns = { run = function(v) return v + 1 end }
print(fns.run(1))
local which = "run"
fns[which] = function(v) return v .. "!" end
print(fns.run("go"))

local counter = {}
for _, w in ipairs({ "a", "b", "a" }) do
  counter[w] = (counter[w] or 0) + 1
end
print(counter.a, counter.b, counter.c)
