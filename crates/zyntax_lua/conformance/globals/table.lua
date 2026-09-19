x = 10
print(_G.x, _G["x"], rawget(_G, "x"))
_G.y = 20
print(y)
local name = "z"
_G[name] = 30
print(z, _G[name])
print(type(_G), _G == _G._G, _G._VERSION)
print(_G.print == print, _G.string == string, _G.string.rep == string.rep)
local count = 0
for k, v in pairs(_G) do
  if k == "x" or k == "y" or k == "z" then count = count + 1 end
end
print(count)
function f(a) return a + 1 end
print(f(1), _G.f(2), _G["f"](3))
local g = _G
g.w = 40
print(w)
print(rawget(_G, "nothere"), nothere)
setmetatable(_G, { __index = function(t, k) return "dflt:" .. k end })
print(undefined_global)
print(rawget(_G, "undefined_global"))
_G.x = nil
print(x)
print(_ENV.y, _ENV == _G)
_ENV.v = 50
print(v)
local t = {}
for k in pairs(_G) do t[#t + 1] = k end
table.sort(t)
print(#t > 20)
print(select("#", _G.print("via table")))
local function loc() return "local" end
print(rawget(_G, "loc"), loc())
