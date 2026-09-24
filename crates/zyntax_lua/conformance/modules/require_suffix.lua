-- a require followed by a field or a call still loads the file with the program
local f = require("lib.setsglobal").f
print(f(), shared_from_setsglobal)
print(require("lib.callable")(21), called_global)
print(require"lib.setsglobal".f())
