-- A class table made again by a jump back is a second table: what is
-- stored in one is not in the other.
local n = 0
local made = {}
::again::
local C = {}
C.__index = C
n = n + 1
made[n] = setmetatable({}, C)
if n < 2 then goto again end
function C:who() return "second" end
print(made[2]:who())
print(pcall(function() return made[1]:who() end))
