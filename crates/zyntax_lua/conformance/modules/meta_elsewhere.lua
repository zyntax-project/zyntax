-- A table the main file builds gets its metatable in a module.
local m = require "lib.attach"
local function mk(v) return {f = v, 1, 2} end
local r = mk(nil)
mk(1)
print(r.f, #r)
m.attach(r)
print(r.f, #r, rawget(r, "f"))
