local C = setmetatable({}, {__concat = function(a, b)
  return (type(a) == "table" and "C" or a) .. "|" .. (type(b) == "table" and "C" or b)
end})
local vals = { "ab", 1, 2.5, -0.0, math.maxinteger, 1e100, "é☃", "", "a\0b", C }
local s = tostring(7)
local function show(...)
  local out = table.pack(...)
  for i = 1, out.n do out[i] = tostring(out[i]):gsub('^.-:(%d+): ', 'L%1: ') end
  print(table.unpack(out, 1, out.n))
end
for i, v in ipairs(vals) do
  local ok, r = pcall(function() return v .. s end)
  show(i, ok, r)
  ok, r = pcall(function() return s .. v end)
  show(i, ok, r)
  ok, r = pcall(function() return (v .. s) .. v end)
  show(i, ok, r, #tostring(r))
end
local missing = {}
for _, f in ipairs({
  function() local t = {} return t.x .. s end,
  function() local t = {} return s .. t.x end,
  function() local t = {} return t .. s end,
  function() return true .. s end,
  function() return s .. print end,
  function() local t = {} return s .. t.name .. s end,
}) do show(pcall(f)) end
local parts = {}
for i = 0, 20 do parts[#parts + 1] = vals[i % 7 + 1] .. tostring(i % 10) end
print(table.concat(parts, " "))
print(#(vals[1] .. s), (vals[1] .. s):upper(), ("x" .. s) == "x7", vals[1] .. s == "ab7")
