-- A chunk's name keeps its bytes in every message that shows it,
-- UTF-8 or not.
print(select(2, load("\255a = 1")))
print(select(2, load("x = \255", "=\254name")))
local ok, e = pcall(load("error(\"z\")", "\253chunk"))
print(ok, e)
print(debug.getinfo(load("return 1", "@\252f")).source)
print(debug.getinfo(load("return 1", "@\252f")).short_src)
print(debug.traceback("m\251", 5))
print(#select(2, load("\255\254")))
