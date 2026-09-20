-- require returns the loader data, searches cpath for its message, and
-- keeps preload apart from the program's own files
local dir = os.tmpname()
os.remove(dir)
local sep = package.config:sub(1, 1)
local function shown(msg) return (msg:gsub(dir:gsub("%p", "%%%0"), "<dir>")) end
local function write(name, text)
  local h = assert(io.open(name, "w"))
  h:write(text)
  h:close()
end
local base = dir .. "_"
write(base .. "m1.lua", "NAME = ...; return {name = ...}")
write(base .. "m2.lua", "return false")
local oldpath, oldcpath = package.path, package.cpath
package.path = base .. "?.lua"
package.cpath = base .. "?.so;" .. base .. "?" .. sep .. "init"
local m, data = require("m1")
print(m.name, data == base .. "m1.lua", NAME)
print(select("#", require("m1")), require("m1") == m)
print(require("m2"), select("#", require("m2")))
print(select("#", require("m2")))
local st, msg = pcall(require, "nothing")
print(st, shown(msg))
package.preload.pl = function (...) return {...} end
local pl, ext = require("pl")
print(pl[1], pl[2], ext)
print(package.preload.m1, type(package.searchers))
local searchers = package.searchers
package.searchers = 3
print(pcall(require, "m3"))
package.searchers = searchers
package.path = {}
print(pcall(require, "m3"))
package.path = base .. "?.lua"
package.cpath = 7
print(shown(select(2, pcall(require, "m3"))))
package.path, package.cpath = oldpath, oldcpath
print(select("#", package.loadlib("donotexist", "f")))
print(package.searchpath("xuxu", "a/?.lua;b/?/init.lua;;"))
os.remove(base .. "m1.lua")
os.remove(base .. "m2.lua")
