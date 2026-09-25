-- Each of \n, \r, \n\r and \r\n is one line break: lines count them
-- once and a long string holds each as \n.
local function lexstring (x, y, n)
  local f = assert(load('return ' .. x ..
            ', require"debug".getinfo(1).currentline', ''))
  local s, l = f()
  print(s == y, l == n, l)
end

lexstring("'abc\\z  \n   efg'", "abcefg", 2)
lexstring("'abc\\z  \n\n\n'", "abc", 4)
lexstring("'\\z  \n\t\f\v\n'",  "", 3)
lexstring("[[\nalo\nalo\n\n]]", "alo\nalo\n\n", 5)
lexstring("[[\nalo\ralo\n\n]]", "alo\nalo\n\n", 5)
lexstring("[[\nalo\ralo\r\n]]", "alo\nalo\n", 4)
lexstring("[[\ralo\n\ralo\r\n]]", "alo\nalo\n", 4)
lexstring("[[alo]\n]alo]]", "alo]\n]alo", 2)

local prog = [[
a = 1        -- a comment
b = 2


x = [=[
hi
]=]
y = "\
hello\r\n\
"
return require"debug".getinfo(1).currentline
]]

for _, n in pairs{"\n", "\r", "\n\r", "\r\n"} do
  local p, nn = string.gsub(prog, "\n", n)
  print(assert(load(p))() == nn, _G.x == "hi\n", _G.y == "\nhello\r\n\n")
end
_G.x, _G.y = nil

-- A syntax error after \r breaks names its line.
print(load("x = 1\rx = 2\r\ry = = 3", "=crlf"))
print(load("x = 1\r\nx = 2\n\ry = = 3", "=crlf"))
-- A raw break in a short string is still an error.
print(load("x = 'a\rb'", "=crlf"))
