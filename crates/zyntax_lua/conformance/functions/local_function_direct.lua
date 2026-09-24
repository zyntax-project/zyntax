-- Calls through locals: one holding a function from its declaration on
-- and never assigned again is that function; any other may change or
-- be nil by the time it is called.

local function step(acc, i)
  return acc + i % 8
end
local sum = 0
for i = 0, 99 do
  sum = step(sum, i)
end
print("local function", sum)

local twice = function(x) return x * 2 end
print("function expression", twice(21), twice(2.5))

local function fact(n)
  if n <= 1 then return 1 end
  return n * fact(n - 1)
end
print("recursive", fact(10))

-- A local function assigned again later calls whatever it holds then.
local function rebound(x) return "first " .. x end
print(rebound(1))
rebound = function(x) return "second " .. x end
print(rebound(2))

-- Assigned in a loop: each call sees the latest value.
local pick = function() return 0 end
for i = 1, 3 do
  pick = function() return i end
  io.write(pick(), " ")
end
print()

-- Assigned in a branch: may still be the first value, or nil.
local maybe = function() return "kept" end
local flag = #arg >= 0
if flag then maybe = function() return "branch" end end
print(maybe())
local gone = function() return "here" end
if flag then gone = nil end
print(pcall(function() return gone() end))

-- A closure from a call that always returns one.
local function make_step(k)
  return function(acc, i) return acc + (i + k) % 8 end
end
local stepk = make_step(3)
local total = 0
for i = 0, 99 do
  total = stepk(total, i)
end
print("closure", total)

-- A call whose result may be nil: calling it raises.
local function maybe_make(n)
  if n > 0 then
    return function() return n end
  end
end
local some = maybe_make(1)
local none = maybe_make(0)
print(some())
print(pcall(function() return none() end))

-- Several names from one declaration.
local inc, dec = make_step(1), make_step(-1)
print(inc(0, 1), dec(0, 1))
local a, b = maybe_make(2)
print(a(), b)

-- Captured by a nested function, which calls it directly.
local function outer()
  local base = 10
  local function add(x) return base + x end
  local function use(n)
    local s = 0
    for i = 1, n do s = add(s) + i end
    return s
  end
  return use(4)
end
print("nested", outer())

-- A global declared once, then shadowed by a local of the same name.
function shadowed() return "global" end
print(shadowed())
do
  local shadowed = function() return "local" end
  print(shadowed())
end
print(shadowed())
