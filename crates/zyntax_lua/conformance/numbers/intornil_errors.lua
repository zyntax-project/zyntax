-- an integer or float read that may be nil: arithmetic and ordering
-- raise for nil as on any value; equality compares it

local t = { 1, 2, nil, 4 }
for i = 1, 4 do
  print(i, pcall(function() return t[i] + 1 end))
end
print(pcall(function() local x = t[3]; return x < 1 end))
print(pcall(function() return nil < 1 end))
print(pcall(function() local x = t[3]; return 1 <= x end))
print(pcall(function() return -t[3] end))
print(pcall(function() return t[3] * 2.5 end))
print(pcall(function() return t[1] // 0 end))
print(t[3] == nil, t[1] == nil, t[1] == 1, t[2] == t[1], t[3] == t[3], t[4] ~= nil, t[1] == 1.0)
print(t[1] + 0.5, t[2] * t[4], t[4] // 3, t[4] % 3, t[1] / 2, t[2] ^ 2, -t[1], t[1] < t[2])

local f = { 1.5, nil, 2.5 }
for i = 1, 3 do
  print(i, pcall(function() return f[i] * 2 end))
end
print(f[2] == nil, f[1] == 1.5, f[1] == f[3], f[3] > f[1], (f[2] or 0) + 1)

local counts = {}
for _, w in ipairs({ 3, 1, 3, 2, 3 }) do counts[w] = (counts[w] or 0) + 1 end
print(counts[1], counts[2], counts[3], counts[4])

local o = { state = true }
o.count = 0
local function bump(x) x.count = x.count + 1 return x.count end
print(bump(o), bump(o), o.count >= 2)
o.count = nil
print(pcall(bump, o))

-- a field whose only store reads it first
local tally = {}
for i = 1, 3 do tally.n = (tally.n or 0) + i end
print(tally.n)
