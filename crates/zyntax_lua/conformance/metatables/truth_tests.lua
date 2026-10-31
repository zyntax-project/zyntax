-- Testing a value for truth hands it to nobody: methods reached
-- through such values are still found where the call stands.
local Node = {}
Node.__index = Node
function Node.new(v, nxt) return setmetatable({v = v, nxt = nxt}, Node) end
function Node:sum()
  if not self.nxt then return self.v end
  return self.v + self.nxt:sum()
end
local list = nil
for i = 1, 5 do list = Node.new(i, list) end
local n, total = list, 0
while n do total = total + n.v; n = n.nxt end
print(list:sum(), total)
local f = function(x) return x * 2 end
if f and not (f == nil) then print(f(21)) end
local t = {a = 1}
repeat t.a = t.a + 1 until not t or t.a > 3
print(t.a)
for k, v in pairs(t) do print(k, v) end
