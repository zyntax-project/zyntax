-- debug.upvalueid gives a light userdata: of type userdata, printed
-- as one, equal and a table key by its identity.
local a, b = 1, 2
local function f () return a, b end
local function g () return a end
local ia = debug.upvalueid(f, 1)
local ib = debug.upvalueid(f, 2)
print(type(ia), (tostring(ia):gsub("0x%x+", "ADDR")))
print(ia == debug.upvalueid(g, 1), ia == ib, rawequal(ia, debug.upvalueid(g, 1)))
local t = {}
t[ia] = "a"
t[ib] = "b"
print(t[debug.upvalueid(g, 1)], t[ib])
print(math.type(ia), io.type(ia))
-- A light userdata is a value, never collected: a weak key stays.
local w = setmetatable({}, {__mode = "k"})
w[ia] = "kept"
collectgarbage()
collectgarbage()
print(w[ia], next(w) == ia)

local function try(f, ...)
  local ok, err = pcall(f, ...)
  print(ok, err)
end
try(debug.setuservalue, ia, {})
try(debug.setuservalue, 1, {})
try(debug.setuservalue, {}, {})
try(debug.setuservalue)
try(debug.setuservalue, ia, {}, "x")
try(function () return ia + 1 end)
try(function () return ia < ia end)
try(function () return #ia end)
try(function () return ia.x end)
try(function () return math.sin(ia) end)
print(debug.getuservalue(ia))
print(select("#", debug.getuservalue(ia)))
print(debug.getuservalue(1, 2))
try(debug.getuservalue, 1, "x")
