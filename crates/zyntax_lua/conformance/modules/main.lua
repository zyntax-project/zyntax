local m = require "lib.mod"
print(m.add(1, 2), m.add(3, 4), m.calls(), m.name)
print(require("lib.mod") == m, package.loaded["lib.mod"] == m)
local d = require("lib.util.deep")
print(d.twice(5), d.tag, m.calls())
print(require "lib.noret", x_from_noret, shared_global)
print(package.loaded["lib.util.deep"] == d)
local e = require "lib.erring"
