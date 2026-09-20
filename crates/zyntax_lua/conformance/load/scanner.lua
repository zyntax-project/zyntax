-- the scanner: form feed and vertical tab are whitespace, \z skips every
-- line break that follows, and nothing follows a chunk's return
local function dostring(x) return assert(load(x), "")() end
dostring("x \v\f = \t\r 'a\0a' \v\f\f")
print(x == 'a\0a', string.len(x))
print(load("y = 1 \v")(), y)
print(load("return 'abc\\z  \n   efg'")())
print(load("return 'abc\\z  \n\n\n'")())
print(#load("return '\\z  \n\t\f\v\n'")())
print("abc\z
        def\z
        ghi\z
       " == 'abcdefghi')
print(load("return;;"), load("return 1;;"), (load("return; x = 1")))
print(load(";;return 2")())
print(load("return;") ~= nil, load("x = 1;;") ~= nil, load(";") ~= nil)
print((load("do return; ; end")), load("do ; return end") ~= nil)
