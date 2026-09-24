-- Weak keys are ephemerons: a value that refers back to its own key
-- does not keep the key alive, and a value is kept while its key is.
local t = setmetatable({}, {__mode = "k"})
local held = {}
local function fill()
  for i = 1, 30 do
    local k = {}
    t[k] = {back = k}
  end
  for i = 1, 30 do
    local k = {}
    held[i] = k
    t[k] = {back = k, i = i}
  end
  -- A chain: each value is the next key.
  local first = {}
  local k = first
  for i = 1, 5 do
    local v = {}
    t[k] = v
    k = v
  end
  held.chain = first
end
fill()
collectgarbage()
collectgarbage()
local n = 0
for k, v in pairs(t) do n = n + 1 end
print("entries", n >= 35 and n < 45)
for i = 1, 30 do assert(t[held[i]].i == i and t[held[i]].back == held[i]) end
local k, len = held.chain, 0
while t[k] do k = t[k]; len = len + 1 end
print("chain", len)
