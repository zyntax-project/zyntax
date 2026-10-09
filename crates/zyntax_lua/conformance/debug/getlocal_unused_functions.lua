-- A program that reads locals sees every local function, used or not.
local function unused_one() return 1 end
local function unused_two() return unused_one() end
local i = 1
while true do
  local name, value = debug.getlocal(1, i)
  if not name then break end
  print(name, type(value))
  i = i + 1
end
