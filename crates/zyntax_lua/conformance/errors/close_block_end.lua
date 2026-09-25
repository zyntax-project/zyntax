-- An error closing a block's variables leaves at once: nothing after
-- the block runs.
local function func2close (f) return setmetatable({}, {__close = f}) end

local function foo ()
  do
    local x1 <close> = func2close(function (_, msg)
      print("x1 sees", msg); error("@Y")
    end)
    local x123 <close> = func2close(function (_, msg)
      print("x123 sees", msg); error("@X")
    end)
  end
  print("not reached")
end
print(pcall(foo))

local function brk ()
  for i = 1, 3 do
    local x <close> = func2close(function () error("in break") end)
    if i == 1 then break end
  end
  print("not reached")
end
print(pcall(brk))

local function loop ()
  for i = 1, 3 do
    local x <close> = func2close(function () error("in pass " .. i) end)
    print("pass", i)
  end
  print("not reached")
end
print(pcall(loop))

local function jump ()
  do
    local x <close> = func2close(function () error("in goto") end)
    goto out
  end
  ::out::
  print("not reached")
end
print(pcall(jump))

local function generic ()
  local function it (_, i) if i < 2 then return i + 1 end end
  for i in it, nil, 0, func2close(function () error("closing value") end) do
    print("step", i)
  end
  print("not reached")
end
print(pcall(generic))
