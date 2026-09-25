-- A generic for's fourth value must be closable.
local function it () return nil end
print(pcall(function ()
  for k in it, nil, nil, 42 do end
end))
print(pcall(function ()
  local t = {}
  for k in it, nil, nil, t do end
end))
local ran = false
print(pcall(function ()
  for k in function () ran = true end, nil, nil, {} do end
end))
print("iterator ran", ran)
