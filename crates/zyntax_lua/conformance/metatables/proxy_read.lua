-- A metatable whose __index is another table: fields read through it.
local Methods = {}
function Methods:get() return self.x * 2 end
function Methods:tag() return "m" end
local mt = {__index = Methods}
local p = setmetatable({x = 21}, mt)
print(p.x, rawget(p, "get") == nil)
print(type(p.get), p.get ~= nil)
local h = p.get
print(h(p))
print(p:get(), p.tag(p))
print(p.missing == nil, mt.get == nil)
-- An __index function answers a field the constructors leave nil.
local function mk(v) return {x = v, y = 1} end
local t = mk(nil); mk(5)
setmetatable(t, {__index = function(_, k) return "fn " .. k end})
print(t.x, t.y)
-- A class whose __index becomes a function after its objects exist.
local Tree = {}
Tree.__index = Tree
local function node(l, r) return setmetatable({left = l, right = r}, Tree) end
local leaf = node(nil, nil)
local n = node(nil, nil)
getmetatable(leaf).__index = function(_, k) return "fn:" .. k end
print(n.right, n.right == nil)
