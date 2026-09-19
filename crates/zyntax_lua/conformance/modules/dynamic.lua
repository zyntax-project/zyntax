-- require with a name only known when the program runs
local names = {"lib.mod", "lib.util.deep"}
for _, n in ipairs(names) do
  local m = require(n)
  print(n, type(m), package.loaded[n] == m)
end
local name = "lib." .. "noret"
print(require(name), package.loaded[name])
local ok, msg = pcall(require, "lib.does_not_exist")
print(ok, msg:match("^module 'lib.does_not_exist' not found:\n\tno field package.preload%['lib.does_not_exist'%]\n\tno file") ~= nil)
print(msg:match("\n\tno file './lib/does_not_exist.lua'\n\tno file './lib/does_not_exist/init.lua'") ~= nil)
print(package.searchpath("lib.mod", package.path))
print(package.searchpath("nowhere", "./?.lua;./?/init.lua"))
print(package.searchpath("lib.mod", "./?.lua", ".", "/"))
print(package.searchpath("lib_mod", "./?.lua", "_", "/"))
package.path = "./lib/?.lua;" .. package.path
print(require("mod").name, package.loaded.mod == require("mod"))
package.path = 5
print(select(2, pcall(require, "anything")):match("no file '5'") ~= nil)
package.path = {}
print(pcall(require, "anything"))
package.path = "./?.lua"
print(pcall(require, "lib.erring"))
print(select(2, pcall(require, "lib.bad_syntax")):match("^error loading module 'lib.bad_syntax' from file './lib/bad_syntax.lua'") ~= nil)
local mod = require "lib.mod"
print(mod.add(2, 3), select("#", require("lib.mod")))
