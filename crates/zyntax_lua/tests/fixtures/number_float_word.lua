local maxi, mini = math.maxinteger, math.mininteger
local function pick(k, a, b) if k % 2 == 0 then return a end return b end

-- results of integer arithmetic fed to float arithmetic
for k = 0, 1 do
  local a, b = pick(k, maxi, 2.5), pick(k, 1, 2.5)
  local s = a + b
  print(math.type(s), s, s + 0.5, s * 1.0, s / 3, s - 0.25)
  local m = pick(k, mini, -1.5)
  local n = -m
  print(math.type(n), n, n + 0.5, n * 2.0)
  print(math.type(math.abs(m)), math.abs(m), math.abs(m) + 0.5)
  local d = pick(k, 7, 7.5)
  print(d // 2, (d // 2) + 0.25, d % 4, (d % 4) + 0.25, -d // 2, (-d % 4) + 0.5)
  print(math.floor(pick(k, 3.7, 3)) + 0.5, math.ceil(pick(k, -3.7, -3)) + 0.5)
  print(math.floor(pick(k, 1e300, 5)) + 0.5, math.floor(pick(k, 2^62, 5)) + 0.5)
end

-- integers held in a Number slot, read back and mixed with floats
local P = {}
P.__index = P
function P.new(x, y) return setmetatable({x = x, y = y}, P) end
local objs = { P.new(0, 0), P.new(1.5, 2.5), P.new(maxi, mini), P.new(3, 4.25) }
for _, o in ipairs(objs) do
  o.x = o.x + 1
  o.y = o.y * 2
  print(math.type(o.x), o.x, o.x + 0.5, o.y, o.y - 0.5, o.x * o.y, o.x - o.y, o.x // 2, o.y % 3)
  local s = o.x
  o.x = 7
  print(o.x + 0.5, s + 0.5, o.x / 2)
end

-- unboxed values
local boxes = { 3, 4.5, maxi, mini, -0.0, 0 }
for _, v in ipairs(boxes) do
  local w = v
  if type(v) == "number" then
    print(math.type(w), w + 0.5, w * 0.5, w - 0.5, w // 1.0, w % 2.0, -w, math.abs(w))
  end
end

-- loops of unknown kind
local function span(a, b, c)
  local t = {}
  for i = a, b, c do t[#t + 1] = i + 0.5 end
  return table.concat(t, ",")
end
print(span(1, 3, 1), span(1.0, 3, 1), span(1, 3, 0.5), span(3, 1, -1))
print(string.format("%.17g %.17g", maxi + 1.0, (maxi + 1) + 0.0))
