-- package.loadlib, and require's C and all-in-one searchers
package.path = "./?.lua"
package.cpath = "./?.so"

local function show(...)
  local out = {}
  for i = 1, select("#", ...) do
    local v = select(i, ...)
    if type(v) == "string" and v:find("\n") then v = v:match("^(.-)\n") end
    out[#out + 1] = type(v) == "function" and "function" or tostring(v)
  end
  print(table.concat(out, " "))
end

-- loadlib: "*" opens the library alone; a name gives that C function
show(package.loadlib("./cload.so", "*"))
local f = package.loadlib("./cload.so", "cload_twice")
print(type(f), f(21), f == package.loadlib("./cload.so", "cload_twice"))
local fn, msg, where = package.loadlib("./cload.so", "no_such_function")
print(fn, type(msg), where)
fn, msg, where = package.loadlib("./no-such-library.so", "f")
print(fn, type(msg), where)
fn, msg, where = package.loadlib("./no-such-library.so", "*")
print(fn, type(msg), where)

-- the C searcher: the loader sees the name and the file
local m, file = require "cload"
print(m.which, m.name, m.file, file)
print(require "cload" == m, select("#", require "cload"))

-- the all-in-one searcher: a submodule's open function in its root's file
local s, sfile = require "cload.sub"
print(s.which, s.name, s.file, sfile)

-- the hyphen: luaopen_ and the part before it
print(require "cload-v2")

-- an open function that raises
print(pcall(require, "cload.bad"))
print(package.loaded["cload.bad"])

-- a library without the open function
local ok, e = pcall(require, "cnoopen")
show(ok, e)

-- the all-in-one searcher finds the file but not the function
ok, e = pcall(require, "cload.missing")
print(ok)
print(e)

-- found nowhere
ok, e = pcall(require, "nowhere.at.all")
print(ok)
print(e)
