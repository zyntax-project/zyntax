local pieces = { "return ", "\x27joined\x27" }
local i = 0
print(load(function() i = i + 1; return pieces[i] end)())
print(select("#", load("")()))
local g, err = load("return +")
print(g, err ~= nil)
