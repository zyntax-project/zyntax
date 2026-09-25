-- the library reached other than by its global's name.
local function mk() local v = 1 return function() return v end end
local f = mk()
print(_G.debug.getupvalue(f, 1))
print(_G.debug.setupvalue(f, 1, "changed"))
print(f())
print(package.loaded.debug.setupvalue(f, 1, 3))
print(f())
local D = _G["deb" .. "ug"]
print(D.getupvalue(f, 1))
local function g() D.setlocal(2, 1, 99) end
local function k()
  local y = 1
  g()
  return y
end
print(k())
print(D.traceback("t"))
