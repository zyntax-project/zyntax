-- values C holds only on its stack, in upvalues or in the registry
-- survive the collections the program's allocations cause
package.cpath = "./?.so"
local m = require "cgc"

local function churn()
  local junk = {}
  for i = 1, 20000 do
    junk[i % 100 + 1] = { i, tostring(i) .. "x" }
  end
  collectgarbage()
end

print(m.hold(churn, 200))
local ref = m.stash()
churn()
print(m.unstash(ref))
local read = m.reader()
churn()
print(read())
print(m.payload(churn))
for i = 1, 3 do
  print(m.hold(churn, 50))
end
