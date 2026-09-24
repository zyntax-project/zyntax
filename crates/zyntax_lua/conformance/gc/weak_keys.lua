-- A weak-key table loses the entries whose keys are collected, and
-- keeps those whose keys are held elsewhere or are not objects.
local t = setmetatable({}, {__mode = "k"})
local held = {}
local function fill(n)
  for i = 1, n do
    held[i] = {}
    t[held[i]] = i
    t[{}] = -i
  end
  t.name = "kept"
  t[42] = "number"
  t[true] = "boolean"
end
fill(100)
collectgarbage()
collectgarbage()
local positives, negatives = 0, 0
for k, v in pairs(t) do
  if type(k) == "table" then
    if v > 0 then positives = positives + 1 else negatives = negatives + 1 end
  end
end
print("held keys", positives)
print("dropped most", negatives < 10)
print(t.name, t[42], t[true])
for i = 1, 100 do assert(t[held[i]] == i) end
print(getmetatable(t).__mode)
