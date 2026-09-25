-- table.unpack refuses as many results as the reference's stack
-- cannot hold.
print(pcall(table.unpack, {}, 1, 1e6))
print(select('#', table.unpack({}, 1, 100)))

local function f ()
  for i = 999900, 1000000, 1 do table.unpack({}, 1, i) end
end
local ok, msg = pcall(f)
print(ok, msg)

print(pcall(table.unpack, {}, 0, math.maxinteger))
print(pcall(table.unpack, {}, math.mininteger, math.maxinteger))
print(select('#', table.unpack({}, math.maxinteger, math.mininteger)))
