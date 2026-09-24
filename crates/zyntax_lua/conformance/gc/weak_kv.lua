-- An all-weak table loses an entry when either its key or its value
-- is collected.
local t = setmetatable({}, {__mode = "kv"})
local keys, values = {}, {}
local function fill()
  for i = 1, 50 do
    local k, v = {}, {}
    keys[i] = k
    t[k] = v
    if i <= 10 then values[i] = v end
  end
  for i = 1, 50 do t[{}] = "orphan key" end
  t[1] = {}
end
fill()
collectgarbage()
local both, total = 0, 0
for k, v in pairs(t) do
  total = total + 1
  if type(k) == "table" and type(v) == "table" then both = both + 1 end
end
print("pairs with both held", both)
print("few others", total - both < 10)
for i = 1, 10 do assert(t[keys[i]] == values[i]) end
print(rawlen(t) <= 1)
