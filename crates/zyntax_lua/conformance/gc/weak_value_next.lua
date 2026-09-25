-- A traversal of a weak-value table yields no nil value, even when a
-- collection clears values while it runs, and keeps every held entry.
local t = setmetatable({}, {__mode = "v"})
local held = {}
local function fill()
  for i = 1, 600 do
    local v = {i}
    t[i] = v
    t["k" .. i] = {i}
    if i % 3 == 0 then held[#held + 1] = v end
  end
end
fill()
local nils = 0
local k, v = next(t)
while k ~= nil do
  if v == nil then nils = nils + 1 end
  local _ = {k}
  k, v = next(t, k)
end
for key, value in pairs(t) do
  if value == nil then nils = nils + 1 end
  local _ = tostring(key) .. "!"
end
local kept = 0
for _, value in ipairs(held) do
  if t[value[1]] == value then kept = kept + 1 end
end
print("nil values", nils)
print("held entries", kept)
