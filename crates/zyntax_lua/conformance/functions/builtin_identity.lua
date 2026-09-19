-- a builtin is one value, whatever names it
print(next == next, print == print, pairs({}) == next, string.len == string.len, ("x").len == string.len)
local a, b = math.sin, math.sin
print(a == b, rawequal(a, b))
print(ipairs{} == ipairs{}, pairs({}) == next, select(1, ipairs({})) == select(1, ipairs({})))
