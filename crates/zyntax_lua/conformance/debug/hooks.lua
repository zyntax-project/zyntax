-- debug.sethook and gethook: line, call, return and count hooks.
local lines = {}
local function hook(event, line)
  if event == "line" then
    lines[#lines + 1] = line
  else
    lines[#lines + 1] = event:sub(1, 1) .. debug.getinfo(2, "S").linedefined
  end
end
local function run(name, f, mask, count)
  lines = {}
  debug.sethook(hook, mask, count)
  f()
  debug.sethook()
  print(name, table.concat(lines, " "))
end

run("straight", function()
  local a = 1
  local b = 2
  a = a + b
end, "l")

run("numeric for", function()
  local s = 0
  for i = 1, 3 do
    s = s + i
  end
  s = s + 1
end, "l")

run("while", function()
  local i = 0
  while i < 3 do
    i = i + 1
  end
  i = 0
end, "l")

run("repeat", function()
  local i = 0
  repeat
    i = i + 1
  until i >= 2
  i = 0
end, "l")

run("if", function()
  local x = 1
  if x > 0 then
    x = 2
  else
    x = 3
  end
  x = 4
end, "l")

run("calls", function()
  local function g()
    return 1
  end
  local z = g()
  z = z + 1
end, "l")

run("call and return", function()
  local function g()
    local q = 1
  end
  g()
end, "cr")

print(debug.gethook())
debug.sethook(hook, "crl", 3)
local h, mask, count = debug.gethook()
debug.sethook()
print(h == hook, mask, count)
print(debug.gethook())

local n = 0
debug.sethook(function() n = n + 1 end, "", 1)
local x = 0
for i = 1, 10 do x = x + i end
debug.sethook()
print(n > 0)
