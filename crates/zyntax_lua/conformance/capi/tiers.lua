-- a C function called from code that runs interpreted, then compiled
package.cpath = "./?.so"
local m = require "ctiers"
local add, pair = m.add, m.pair

local function loop(n)
  local s = 0
  for i = 1, n do
    s = add(s, i)
    local a, b = pair(i, s)
    if a ~= s or b ~= i then error("pair") end
  end
  return s
end

print(loop(10))
print(loop(1000))
print(loop(200000))
local f = add
print(f(1, 2), m.add(40, 2))
print(pcall(add, 1, {}))
