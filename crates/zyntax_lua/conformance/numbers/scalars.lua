local function pick(c) local x if c then x = 5 end return x end     -- Int|Nil
local function flag(c) local f = nil if c then f = true end return f end -- Bool|Nil
local function mix(c) local v = 1 if c == 1 then v = 2.5 elseif c == 2 then v = false elseif c == 3 then v = nil end return v end
local a, b = pick(true), pick(false)
print(a, b, a == 5, b == nil, a == b, b == b, a ~= nil)
print(flag(true), flag(false), flag(true) == true, flag(false) == false, flag(false) == nil)
print(mix(0), mix(1), mix(2), mix(3), mix(1) == 2.5, mix(2) == false, mix(3) == nil, mix(0) == 1)
print(a + 1, a * 2.5, a // 2, a % 3, a / 2, a ^ 2, -a, a < 6, a <= 5, a > 4, 6 > a)
print(pcall(function() return b + 1 end))
print(pcall(function() return b < 1 end))
print(pcall(function() return -b end))
print(pcall(function() return flag(true) + 1 end))
print(pcall(function() return mix(2) * 2 end))
local count = nil
for i = 1, 5 do count = (count or 0) + i end
print(count, count == 15, count > 10)
local s = 0
for i = 1, 10 do local v = pick(i % 2 == 0) if v then s = s + v end if not v then s = s + 100 end end
print(s)
local t = {}
t[a] = "five"; t[b or 0] = "zero"
print(t[5], t[0])
print(tostring(a), tostring(b), type(a), type(b), tostring(flag(true)), type(flag(false)))
print(a .. "", pcall(function() return b .. "" end))
print(math.abs(-a), math.max(a, 2), math.type(a), math.type(b))
print(string.format("%s %s %d", a, b, a), select("#", a, b))
local z = mix(1)
z = z + 1
print(z, mix(3) or "dflt", mix(2) or "dflt2", mix(2) and "yes" or "no", not mix(3), not mix(2), not mix(0))
if mix(3) then print("wrong") else print("nil is false") end
if mix(2) then print("wrong") else print("false is false") end
if mix(0) then print("1 is true") end
