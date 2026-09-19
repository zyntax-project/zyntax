-- %p: an address for what is allocated, (null) for the rest
print(string.format("%p", 1), string.format("%p", nil), string.format("%p", true), string.format("%p", 1.5))
local t = {}
print(string.format("%p", t):match("^0x%x+$") ~= nil, string.format("%p", t) == string.format("%p", t))
print(string.format("%p", print):match("^0x%x+$") ~= nil, string.format("%p", {}) ~= string.format("%p", {}))
print(string.format("%p", "abc"):match("^0x%x+$") ~= nil, string.format("%p", coroutine.create(print)):match("^0x") ~= nil)
print(string.format("%10p|", 1), string.format("%-8p|", nil))
