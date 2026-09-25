-- the C API's stack, values, tables, upvalues, registry and userdata
package.cpath = "./?.so"
local m = require "cbasics"

print(m.types(1, 2.5, "s", nil, true, {}, print, m.types))
print(m.types())
print(m.shuffle())
print(m.fstrings())
print(m.convert(42))
print(m.convert(3.0))
print(m.convert(3.5))
print(m.convert("0x10"))
print(m.convert(" 12 "))
print(m.convert("abc"))
print(m.convert(false))
print(m.tables())
to_c = "hello"
print(m.globals())
print(from_c)
print(m.arith())

local c1 = m.newcounter("tag")
print(c1(), c1(), c1())
local c2 = m.newcounter({})
print(c2() == 1, select(2, c2()) ~= nil)
print(m.same())

local ref = m.keep({ answer = 42 })
print(m.fetch(ref).answer)
local ref2 = m.keep("second")
print(m.fetch(ref2))
m.drop(ref)
print(m.keep("reused") == ref)
print(m.registry())

local b = m.box(20, "red")
print(b.v, b.tag, b.other, tostring(b), b + 5, #b)
print(type(b), m.check(b))
print(m.check({}))
print(m.light())
print(getmetatable(b).__name)
