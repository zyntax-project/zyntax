-- load with a reader function.

local function read1(x)
  local i = 0
  return function ()
    collectgarbage()
    i = i + 1
    return string.sub(x, i, i)
  end
end

-- a reader raising: load returns the error
local f, msg = load(function () error("hhi") end)
print(f, string.find(msg, "hhi") ~= nil)
f, msg = load(function () error({}) end)
print(f, msg ~= nil)

-- a piece that is a number is its string
local n = 0
f = load(function ()
  n = n + 1
  if n == 1 then return "return " elseif n == 2 then return 57 end
end)
print(f())
n = 0
f = load(function ()
  n = n + 1
  if n == 1 then return "return " elseif n == 2 then return 1.5 end
end)
print(f())

-- nil or an empty string ends the chunk
local t = {"return ", "3", "", "+ 1"}
f = load(function () return table.remove(t, 1) end)
print(f())
t = {nil, "return ", "3"}
f = load(function () return table.remove(t, 1) end)
print(f())

-- any other piece is refused
f, msg = load(function () return {} end)
print(f, string.find(msg, "reader function must return a string", 1, true) ~= nil)
f, msg = load(function () return true end)
print(f, string.find(msg, "reader function must return a string", 1, true) ~= nil)

-- a long binary chunk one byte at a time
local long = string.dump(function ()
  return '01234567890123456789012345678901234567890123456789'
end)
print(load(read1(long))())
local x = string.dump(load("x = 1; return x"))
local a = assert(load(read1(x), nil, "b"))
print(a(), _G.x)
print(load(read1(x), nil, "t"))
_G.x = nil
x = [[
  return function (x)
    return function (y)
     return function (z)
       return x+y+z
     end
   end
  end
]]
a = assert(load(read1(x), "read", "t"))
print(a()(2)(3)(10))
x = string.dump(a)
a = assert(load(read1(x), "read", "b"))
print(a()(2)(3)(10))
print(load(read1("*a = 123")))
