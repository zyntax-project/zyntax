-- an error from C that nothing catches ends the program as the
-- reference ends it
package.cpath = "./?.so"
local m = require "cerrors"
print("before")
m.boom(3)
print("not reached")
