-- A string past the pool's largest class built every step and left in
-- a global, where only the collector reclaims the one before: each is
-- a block of its own, outside the pool.
local base = string.rep("x", 4096)
held = ""

local function main(n)
  local total = 0
  for i = 1, n do
    held = base .. i
    total = total + #held
  end
  return total
end

print(main(tonumber(arg[1])))
