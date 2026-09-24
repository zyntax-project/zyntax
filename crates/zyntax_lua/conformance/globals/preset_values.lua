-- the globals Lua sets before the program runs, read as values
print(_VERSION)
print(_VERSION == "Lua 5.4", type(_VERSION), _VERSION:match("%d+%.%d+"))
if _VERSION == "Lua 5.1" then print("old") else print("new") end
local v = _VERSION
print(v, #v)
local s = string
if s then print("library table") else print("nil") end
print(s == nil, type(s), s.format("%d", 7), s == string)
local a = arg
print(a == nil, type(a))
if math then print(math.type(1)) end
