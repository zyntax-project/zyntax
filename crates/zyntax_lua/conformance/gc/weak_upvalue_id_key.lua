-- An upvalue's identity kept as a key of a weak-value table stays when
-- the table drops the entries whose values were collected.
local x = 1
local function g() return x end
local id = debug.upvalueid(g, 1)
local keep = {}
local t = setmetatable({}, { __mode = "v" })
t[id] = keep
for i = 1, 10 do t["k" .. i] = {} end
collectgarbage()
collectgarbage()
t.fresh = keep
print(t[id] == keep)
local found = false
for k in pairs(t) do found = found or k == id end
print(found)
