-- A local a closure shares, set by setlocal from a frame above it, is
-- seen by the closure at once; one the closure changes is seen by
-- getlocal at once.
local function f()
  local b = 2
  local function inner()
    debug.setlocal(2, 1, "B")
    return b
  end
  print(inner(), b)
  local c = 1
  local function bump()
    c = c + 10
    return select(2, debug.getlocal(2, 3))
  end
  print(bump(), c)
  local n = 0
  local function count()
    n = n + 1
    return n
  end
  local function reset()
    debug.setlocal(2, 5, 100)
  end
  count()
  reset()
  print(count(), n)
end
f()

-- The same with the main chunk's locals.
local top = "t"
local function peek() return top end
local function poke()
  debug.setlocal(2, 2, "T")
  return peek()
end
print(poke(), top, peek())

-- Loops: each iteration's local is its own.
local fs = {}
for i = 1, 3 do
  local v = i
  fs[i] = function() return v end
  local function set() debug.setlocal(2, 10, v * 100) end
  set()
end
print(fs[1](), fs[2](), fs[3]())
