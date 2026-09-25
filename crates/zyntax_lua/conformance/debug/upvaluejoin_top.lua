-- upvaluejoin and setupvalue on locals of the main chunk.
local a, b = 1, 2
local function fa() return a end
local function fb() return b end
print(fa(), fb())
debug.upvaluejoin(fb, 1, fa, 1)
print(fa(), fb())
a = 10
print(fa(), fb(), b)
print(debug.upvalueid(fb, 1) == debug.upvalueid(fa, 1))

local pt = {y = 8}
local cnt = 6
local function inc() cnt = cnt + 1 return cnt end
local g = function() return pt.y end
print(pcall(g))
debug.upvaluejoin(g, 1, inc, 1)
print(pcall(g))
print(inc(), cnt)

local c = 0
local function counter() c = c + 1 return c end
counter()
print(debug.setupvalue(counter, 1, 100))
print(counter(), c)
print(debug.getupvalue(counter, 1))

function global_reader() return a + b end
print(global_reader())
debug.setupvalue(global_reader, 2, 5)
print(global_reader(), b)
