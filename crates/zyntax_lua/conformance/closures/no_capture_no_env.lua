-- Closures that capture nothing, called directly and as values.

local function main()
  local step = function(acc, i) return acc + i % 8 end
  local sum = 0
  for i = 0, 999 do
    sum = step(sum, i)
  end
  return sum
end
print(main())

-- Stored in a table and called through it.
local ops = {
  add = function(a, b) return a + b end,
  mul = function(a, b) return a * b end,
}
print(ops.add(2, 3), ops.mul(4, 5))

-- Passed to the library.
local t = { 5, 3, 9, 1, 7 }
table.sort(t, function(x, y) return x > y end)
print(table.concat(t, " "))
print((string.gsub("abc", "%w", function(c) return c:upper() .. "." end)))
print(pcall(function(x) return x + 1 end, 41))

-- Returned without capturing: each call makes a new value.
local function maker()
  return function(x) return -x end
end
local n1, n2 = maker(), maker()
print(n1(3), n2(4), n1 == n2)

-- A nested function whose inner function captures from two levels up:
-- the middle one carries the capture for it.
local function outer(k)
  local function middle()
    local function inner(x) return x + k end
    return inner(1)
  end
  return middle()
end
print(outer(10))

-- One capturing nothing, called in a loop from a function that does.
local function scale(v)
  local double = function(x) return x * 2 end
  local s = 0
  for i = 1, 5 do s = s + double(i) * v end
  return s
end
print(scale(3))

-- Variadic, capturing nothing.
local count = function(...) return select("#", ...) end
print(count(), count(1, nil, 3))

-- Called with too few and too many arguments.
local pair = function(a, b) return tostring(a) .. "," .. tostring(b) end
print(pair(1), pair(1, 2, 3))

-- Used as a method.
local obj = { name = "obj" }
obj.greet = function(self, who) return self.name .. " greets " .. who end
print(obj:greet("you"))
