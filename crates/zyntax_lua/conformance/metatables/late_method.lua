-- A method called before it exists raises; once assigned, it is found.
local T = {}; T.__index = T
local o = setmetatable({v = 3}, T)
for i = 1, 3 do
  local ok, err = pcall(function() return o:late(i) end)
  print(ok, err)
  if i == 2 then
    function T:late(x) return self.v + x end
  end
end
print(o:late(10))
T.late = nil
print(pcall(function() return o:late(1) end))
