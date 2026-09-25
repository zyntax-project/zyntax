-- getlocal and setlocal called in tail position still see the frame
-- that calls them.
local function d(x)
  local y = x * 2
  return debug.getlocal(1, 2)
end
print(d(3))

local function s(x)
  local y = x * 2
  return debug.setlocal(1, 2, 5)
end
print(s(3))

local dbg = debug
local function a(x)
  local y = x + 1
  return dbg.getlocal(1, 1)
end
print(a(10))

local function both(p, q)
  local r = p .. q
  return debug.getlocal(1, 3), debug.getlocal(1, 1)
end
print(both("a", "b"))

local function va(...)
  local k = select("#", ...)
  return debug.getlocal(1, -2)
end
print(va("u", "v", "w"))

local function nested(x)
  local y = x
  local function inner()
    return debug.getlocal(2, 2)
  end
  local n, v = inner()
  return n, v
end
print(nested(7))

local function set_above(x)
  local y = x
  local function inner()
    return debug.setlocal(2, 2, "set")
  end
  local n = inner()
  return n, y
end
print(set_above(1))
