-- a function statement assigning a local declared earlier: its parameters take anything
local f
f = function (a)
  local x = "q"
  x = a
  print(x, type(x))
end
f(2)
local g
function g(a)
  local y = "q"
  y = a
  print(y, type(y))
end
g(2)
local h
function h(a)
  local z = 5
  z = a
  print(z, type(z))
end
h("s")
local k
function k(a)
  local w = "q"
  w = a
  print(w, type(w))
end
k(2)
k("t")
