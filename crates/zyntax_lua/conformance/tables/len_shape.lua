-- the length of shaped tables and strings

-- an array part, grown, with a hole at the end and in the middle
local list = {name = "list", 1, 2, 3}
print(#list)
list[#list + 1] = 4
list[#list + 1] = 5
print(#list)
list[#list] = nil
print(#list)
local hole = {tag = 1, 10, 20, 30, 40}
hole[2] = nil
print(#hole == 4 or #hole == 1)
local empty = {tag = 1}
print(#empty)
empty[1] = "a"
print(#empty)

-- a shape whose class has a length handler (its name is built here, so
-- that the length of a table no class handles is read in place)
local LEN = "__" .. "len"
local Sized = {}
Sized.__index = Sized
Sized[LEN] = function(t) return t.n * 2 end
local s = setmetatable({n = 21}, Sized)
print(#s)
s.n = 5
print(#s)

-- a shape with a class that has no length handler
local Plain = {}
Plain.__index = Plain
local p = setmetatable({v = 1, 7, 8, 9}, Plain)
print(#p)

-- a class given a length handler later
local Late = {}
Late.__index = Late
local l = setmetatable({v = 1, 1, 2}, Late)
print(#l)
Late[LEN] = function() return 99 end
print(#l)

-- strings
local str = "hello"
print(#str, #"", #(str .. " world"))
local function length(x) return #x end
print(length("abc"), length({tag = 1, 1, 2}))
print(pcall(length, nil))
