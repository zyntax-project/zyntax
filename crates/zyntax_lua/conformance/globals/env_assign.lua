-- _ENV assigned: the chunk's environment from then on
x = 1
_ENV = setmetatable({}, {__index = _G})
y = 2
print(x, y, _G.y, rawget(_ENV, "y"), _ENV ~= _G)
local function f() return z end
z = 3
print(f(), _G.z)
_ENV = _G
print(y, _ENV == _G)
local g = load("_ENV = {p = 5}; return p, _ENV.p")
print(g())
local h = load("return _ENV", "c", "t", {q = 7})
print(h().q)
local pr, pc = print, pcall
_ENV = nil
pr(pc(function() return w end))
