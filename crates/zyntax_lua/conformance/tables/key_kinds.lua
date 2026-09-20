-- keys of every kind, apart: false is not 0, true is not 1, 1 is 1.0
local k = {}
k[0] = "zero"; k[false] = "false"; k[1] = "one"; k[true] = "true"; k[1.0] = "float one"
print(k[0], k[false], k[1], k[true], k[1.0])
k[2^53] = "big"; k[10e30] = "huge"; k[-10e30] = "minus huge"; k[1 / 0] = "inf"; k[-1 / 0] = "-inf"; k[0.5] = "half"
print(k[2^53], k[10e30], k[-10e30], k[1 / 0], k[-1 / 0], k[0.5])
local big = {}
for v = 3000, -3000, -1 do big[v + 0.0] = v end
big[10e30] = "alo"; big[true] = 10; big[false] = 20
print(big[10e30], big[not 1], big[10 < 20], big[0], big[-3000], big[3000])
local bad = 0
for v = 3000, -3000, -1 do if big[v] ~= v then bad = bad + 1 end end
print(bad)
print(pcall(function () local t = {}; t[0 / 0] = 1 end))
print(pcall(function () local t = {}; t[nil] = 1 end))
print(({[1] = "a"})[1.0], ({[1.0] = "b"})[1], ({[2^53] = "c"})[2^53 + 0.0])
