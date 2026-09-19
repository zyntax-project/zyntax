local s = require "string"
print(s == string, require("math") == math, require"_G" == _G)
print(package.loaded.string == string, package.loaded._G == _G, type(package.path))
local debug = require "debug"
print(debug.sethook(), debug.gethook())
local smt = getmetatable("")
print(type(smt), smt.__index == string, ("x"):rep(2))
smt.__index = function(s, k) return "custom:" .. k end
print(("x").foo)
smt.__index = string
print(("abc"):upper())
local t = setmetatable({}, { __metatable = "locked" })
print(getmetatable(t))
print(pcall(setmetatable, t, {}))
print(debug.getmetatable(t).__metatable)
print(_G == _G, require"_G" == _G, rawequal(require"_G", _G), package.loaded._G == _G)
