-- Local functions nothing that runs refers to, beside ones that are.
local function unused(x) return x + 1 end

local function used(x) return x * 2 end
print("used", used(21))

-- Reached only from a function that is itself never reached.
local function countdown(n)
  if n > 0 then return countdown(n - 1) end
  return 0
end
local function never(n) return countdown(n) end

-- Reached from a function that runs.
local function base() return 1 end
local function next_up() return base() + 1 end
print("chain", next_up())

-- Reached from a nested function of a function that runs.
local function deep() return "deep" end
local function mid()
  local function leaf() return deep() end
  return leaf()
end
print("nested", mid())

-- A function that never runs leaves what it would write alone.
local counter = 0
local function bump()
  counter = counter + 1
  never_set = true
end
counter = counter + 5
print("counter", counter, never_set)

-- Shadowed: the first is never named again.
local function pick() return "first" end
local function pick() return "second" end
print("shadow", pick())

-- Declared on every iteration, never called.
local total = 0
for i = 1, 3 do
  local function tmp() return i end
  total = total + i
end
print("loop", total)

-- A function holding unused ones of its own.
local function host()
  local function a() return "a" end
  local function b() return a() end
  return "host"
end
print("host", host())

-- A loaded chunk with unused local functions.
local chunk = load("local function u() return 1 end local function w() return 2 end return w()")
print("loaded", chunk())
