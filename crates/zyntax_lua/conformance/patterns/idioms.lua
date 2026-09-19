-- split
local function split(s, sep)
  local out = {}
  for piece in (s .. sep):gmatch("(.-)" .. sep:gsub("%p", "%%%0")) do
    out[#out + 1] = piece
  end
  return out
end
local parts = split("a,b,,c", ",")
print(#parts, table.concat(parts, "|"))

-- trim
local function trim(s) return (s:gsub("^%s+", ""):gsub("%s+$", "")) end
print("[" .. trim("  padded\t ") .. "]")

-- template expansion with a table and with a function
local tpl = "Hello ${name}, you are ${age}"
print((tpl:gsub("%${(%w+)}", { name = "Ada", age = 36 })))
print((tpl:gsub("%${(%w+)}", function(k) return k:upper() end)))

-- counting and positions
local text = "the quick brown fox jumps over the lazy dog"
local _, spaces = text:gsub(" ", " ")
print(spaces)
local words = 0
for _ in text:gmatch("%a+") do words = words + 1 end
print(words)
print(text:find("fox"))
print(text:find("the", 2))
print(text:match("(%a+)$"))
print(select("#", text:find("(o)")))

-- balanced and frontier
print(("f(x, g(y)) + h(z)"):match("%b()"))
for w in ("THE (quick) BROWN fox"):gmatch("%f[%a]%u+%f[%A]") do io.write(w, " ") end
print()

-- classes and sets
print(("a1 b2 c3"):gsub("%d", "#"))
print(("Hello World"):gsub("%u", "_%0"))
print(("x-y_z"):match("[%w_]+"))
print(("[tag]"):match("%[(.-)%]"))
print(("2024-01-15T10:30"):match("^(%d%d%d%d)%-(%d%d)%-(%d%d)T(%d+):(%d+)$"))
print(("  42  "):match("^%s*(%-?%d+)%s*$"))
print(("abc"):find("b", 1, true))
print(("a.b.c"):gsub("%.", "/"))
print(("a.b.c"):gsub(".", "/"))
print(("hello"):gsub("", "."))
print(("hello"):match(".-(l+)(.*)"))
print(("hello"):gsub("l*", "X"))
print(("key = value # comment"):match("^(%w+)%s*=%s*(.-)%s*#"))

-- position captures and back references
print(("hello world"):find("()o()"))
print(("abcabc"):match("(abc)%1"))
print(("xyzxyz"):gsub("(x)(y)(z)%1", "<%3%2%1>"))

-- gmatch with init and anchors treated literally
for w in ("one two three"):gmatch("%a+", 5) do io.write(w, ",") end
print()
print(("^a^b"):gsub("%^", "!"))

-- byte strings
local s = "\0mid\0"
print(#s, s:find("mid"), s:byte(1), (s:gsub("%z", "0")))
print(("caf\xc3\xa9"):match("[\x80-\xff]+"):byte(1, -1))
print(("%d"):rep(3, "-"))
print(("ab"):rep(0) == "")
