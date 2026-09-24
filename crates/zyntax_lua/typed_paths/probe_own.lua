-- A method held in the instance's own field, and one inherited.
local Base = {}
Base.__index = Base
function Base:twice(x) return x * 2 end
local o = setmetatable({}, Base)
o.step = function(self, x) return x + 1 end
local total = 0
for i = 1, 3 do
  total = total + o:step(i) + o:twice(i)
end
print(total)
