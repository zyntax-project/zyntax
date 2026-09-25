-- A binary chunk starts with the reference's header; a corrupted or
-- truncated chunk is refused.

local header = string.pack("c4BBc6BBB",
  "\27Lua", 0x54, 0, "\x19\x93\r\n\x1a\n", 4,
  string.packsize("j"), string.packsize("n"))

local c = string.dump(function ()
  local a = 1; local b = 3;
  local f = function () return a + b + _ENV.c; end
  local s1 = "a constant"
  local s2 = "another constant"
  return a + b * 3
end)

print(assert(load(c))())
print(string.sub(c, 1, #header) == header)
local ci, cn = string.unpack("jn", c, #header + 1)
print(ci == 0x5678, cn == 370.5)

-- each header byte corrupted
for i = 1, #header do
  local s = string.sub(c, 1, i - 1) ..
            string.char((string.byte(c, i) + 1) % 256) ..
            string.sub(c, i + 1, -1)
  local f, msg = load(s)
  print(i, f, i > 1 and msg or "")
end
-- the integer and the float that follow
local badint = string.sub(c, 1, #header) .. string.pack("j", 0x5679) ..
               string.sub(c, #header + 9)
print(load(badint))
local badnum = string.sub(c, 1, #header + 8) .. string.pack("n", 370.25) ..
               string.sub(c, #header + 17)
print(load(badnum))

-- each truncation
local all = true
for i = 1, #c - 1 do
  local st, msg = load(string.sub(c, 1, i))
  if st or not string.find(msg, "truncated") then
    all = false
    print("not refused at", i, msg)
  end
end
print("every truncation refused", all)
print(load(string.sub(c, 1, 3)))
print(load(string.sub(c, 1, 3), "=chunk"))
print(load(string.sub(c, 1, 3), "@file.luac"))
