-- math functions on arguments whose types are known
local mi, ma = math.mininteger, math.maxinteger
print(math.abs(-7), math.abs(7), math.abs(mi), math.abs(ma))
print(math.abs(-7.5), math.abs(7.5), math.abs(-0.0), math.abs(0/0) ~= math.abs(0/0))
print(math.abs(-1/0), math.abs(1/0))
print(math.floor(3), math.ceil(3), math.floor(-3), math.ceil(-3))
print(math.floor(3.7), math.ceil(3.2), math.floor(-3.7), math.ceil(-3.2))
print(math.type(math.floor(3.7)), math.type(math.ceil(3.2)))
print(math.floor(2^70), math.ceil(-2^70), math.type(math.floor(2^70)))
print(math.floor(-0.0), math.ceil(-0.0), math.floor(0.5), math.ceil(-0.5))
print(math.max(3, 9, 4), math.min(3, 9, 4), math.max(-2), math.min(mi, ma))
print(math.max(3.5, 9.25, 4.0), math.min(3.5, 9.25, 4.0), math.max(2.5), math.min(-0.0, 0.0))
print(math.type(math.max(1, 2)), math.type(math.max(1.0, 2.0)), math.type(math.max(1, 2.5)))
print(math.max(1, 2.5), math.min(1, 2.5), math.max(2.5, 1), math.min(2.5, 1))
local nan = 0/0
print(math.max(nan, 1) ~= math.max(nan, 1), math.max(1, nan), math.min(nan, 1) ~= math.min(nan, 1), math.min(1, nan))
print(math.fmod(7, 3), math.fmod(-7, 3), math.fmod(7, -3), math.fmod(-7, -3))
print(math.fmod(mi, -1), math.fmod(ma, -1), math.fmod(mi, 1), math.fmod(5, 5))
print(math.fmod(7.5, 2), math.fmod(-7.5, 2), math.fmod(7, 2.5), math.fmod(-7.5, -2.0))
print(math.type(math.fmod(7, 3)), math.type(math.fmod(7.0, 3)), math.type(math.fmod(7, 3.0)))
print(math.fmod(1, 1/0), math.fmod(-1, 1/0), math.fmod(1, 0.0) ~= math.fmod(1, 0.0))
print(pcall(function() return math.fmod(5, 0) end))
local function two() return 3, 12 end
print(math.max(two()), math.min(two()), math.max(1, two()), math.min(20, two()))
print(math.max(table.unpack({4, 2, 8})), math.min(table.unpack({4, 2, 8})))
local s, f = 0, 0.0
for i = 1, 10 do
  s = s + math.abs(-i) + math.max(i, 5) + math.min(i, 5) + math.floor(i) + math.fmod(i, 3)
  f = f + math.abs(-i * 0.5) + math.max(i * 0.5, 2.5) + math.fmod(i * 0.5, 2.0)
end
print(s, f, math.type(s), math.type(f))
local t = {}
for i = 1, 5 do t[math.floor(i / 1.5) + 1] = i end
print(#t, t[1], t[2], t[3], t[4])
