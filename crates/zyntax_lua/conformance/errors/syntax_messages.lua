-- Syntax errors as load reports them: one line, the chunk's name and
-- line, Lua's wording, and the token the reader stopped at.

local malformed = {
  "x = ",
  "x = = 1",
  "local = 1",
  "x 1",
  "f() = 1",
  "(f)",
  "local x = 1 local y = x x",
  "function f() return 1",
  "function f()\n  return 1\n",
  "if x then\n  y()\n",
  "while true do\n\n",
  "if x then else elseif y then end",
  "for i = 1 do end",
  "for x do end",
  "repeat x = 1",
  "do local x = 1 end x() end",
  "x = }",
  "x = (1",
  "x = (1\n+ 2\n",
  "t = {1, 2",
  "a.b:c = 1",
  "local function (x) end",
  "function f(1) end",
  "function f() return ... end",
  "return;;",
  "return 1 x = 2",
  "x = 1 +\n\n",
  "x = 'abc",
  "x = 'abc\ny'",
  "x = [[abc",
  "--[[ comment",
  "x = [==[ x ]=]",
  "x = [=x",
  "x = 3x",
  "x = 0x",
  "x = 1e",
  "x = 1..2",
  "x = 0x1p",
  "x = '\\q'",
  "x = '\\300'",
  "x = '\\xZZ'",
  "x = '\\u123'",
  "x = '\\u{80000000}'",
  "x = '\\u{12'",
  "x = \1",
  "goto f; local x; ::f:: print(x)",
  "do goto f end local x ::f:: print(x)",
  "goto nowhere",
  "local function g() goto out end ::out::",
  "break",
  "if x then break end",
  "::a:: ::a::",
  "::a:: do ::a:: end",
  "local x <const> = 1; x = 2",
  "local x <close> = nil; x = 1",
  "local x <const> = 1; function g() x = 2 end",
  "local x <const> = 1; function x() end",
  "local x <foo> = 1",
  "local a <close>, b <close> = nil, nil",
  "a = f(x" .. string.rep(",x", 260) .. ")",
}

for i, s in ipairs(malformed) do
  print(i, pcall(load, s))
end

-- A chunk's name, as the reference shows it.
print(load("x =", "=mychunk"))
print(load("x =", "@file.lua"))
print(load("x =", "a chunk named for itself"))
print(load("x = 1 +", string.rep("n", 50)))
print(load("x = 1 +\ny +", nil))
print(load("\255a = 1", "=bytes"))
local parts = {"x = "}
print(load(function() return table.remove(parts) end))
print(load("x = ", ""))

-- Chunks the reference takes.
local fine = {
  "while true do break; print('no') end return 1",
  "for i = 1, 2 do if i then break end ::continue:: end",
  "do goto l; local x; ::l:: end",
  "local x <const> = 1; local function g() return x end",
  "local t = {} t.x, t.y = 1, 2",
  ";;; local x = 1;; return x",
  "return",
  "return;",
}
for i, s in ipairs(fine) do
  local f, err = load(s)
  print(i, type(f), err)
end

-- break followed by statements runs as a break.
local seen = {}
for i = 1, 3 do
  seen[#seen + 1] = i
  if i == 2 then
    break
    seen[#seen + 1] = "never"
  end
end
print(table.concat(seen, " "))
