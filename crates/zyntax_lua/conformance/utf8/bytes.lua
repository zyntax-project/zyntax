print(#"\xff\xfe", ("\xff"):byte())
local s = "caf\xc3\xa9"
print(#s, s:sub(4):byte(1, -1))
