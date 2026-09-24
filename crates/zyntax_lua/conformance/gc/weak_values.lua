-- A weak-value table loses the entries whose values are collected,
-- in its array part and its hash part alike.
local t = setmetatable({}, {__mode = "v"})
local held = {}
local function fill(n)
  for i = 1, n do
    local a, b = {}, {}
    t[i] = a
    t["k" .. i] = b
    if i % 2 == 0 then
      held[#held + 1] = a
      held[#held + 1] = b
    end
  end
  t.s = "strings stay"
  t.n = 3.5
end
fill(100)
collectgarbage()
collectgarbage()
local arr, hash = 0, 0
for k, v in pairs(t) do
  if type(v) == "table" then
    if type(k) == "number" then arr = arr + 1 else hash = hash + 1 end
  end
end
print("array values", arr >= 50 and arr < 60)
print("hash values", hash >= 50 and hash < 60)
for i = 2, 100, 2 do assert(t[i] ~= nil and t["k" .. i] ~= nil) end
print(t.s, t.n)
print(#t >= 0)
