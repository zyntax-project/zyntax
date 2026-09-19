local function closable(name)
  return setmetatable({}, { __close = function(self, err)
    print("close", name, err)
  end })
end

do
  local a <close> = closable("a")
  local b <close> = closable("b")
  print("in block")
end
print("after block")

local function f()
  local x <close> = closable("x")
  return "ret"
end
print(f())

for i = 1, 2 do
  local c <close> = closable("c" .. i)
  if i == 2 then break end
  print("iter", i)
end

local function g()
  local y <close> = closable("y")
  error("boom")
end
print(pcall(g))

do
  local n <close> = nil
  local fa <close> = false
  print("nil and false are fine")
end

print(pcall(function() local bad <close> = 42 end))

local function h()
  local z <close> = closable("z")
  do
    local w <close> = closable("w")
    goto out
  end
  ::out::
  print("after goto")
end
h()

local function order()
  local a <close> = closable("first")
  local b <close> = closable("second")
  return (function() print("computing"); return 1 end)()
end
print(order())
