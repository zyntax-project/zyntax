-- A variable whose metatable lost __close raises when its block ends,
-- at the block's end.
local function foo ()
  local xyz <close> = setmetatable({}, {__close = print})
  getmetatable(xyz).__close = nil

end
print(pcall(foo))

local function bar ()
  do
    local xyz <close> = setmetatable({}, {__close = print})
    getmetatable(xyz).__close = nil
  end
  print("not reached")
end
print(pcall(bar))

local function baz ()
  for i = 1, 2 do
    local xyz <close> = setmetatable({}, {__close = print})
    getmetatable(xyz).__close = nil
    if i == 1 then
      break
    end
  end
  print("not reached")
end
print(pcall(baz))

local function num ()
  do
    local xyz <close> = setmetatable({}, {__close = print})
    getmetatable(xyz).__close = 4
    local y = 1
  end
end
print(pcall(num))

-- A loop's closing value, at the loop's end, and the loops' breaks.
local function gen ()
  local c = setmetatable({}, {__close = print})
  local function it (_, i) if i < 2 then return i + 1 end end
  for i in it, nil, 0, c do
    getmetatable(c).__close = nil
    local y = 1

  end
  print("not reached")
end
print(pcall(gen))

local function genbreak ()
  local c = setmetatable({}, {__close = print})
  local function it (_, i) if i < 2 then return i + 1 end end
  for i in it, nil, 0, c do
    getmetatable(c).__close = nil
    if i then
      break
    end

  end
  print("not reached")
end
print(pcall(genbreak))

local function rep ()
  repeat
    local xyz <close> = setmetatable({}, {__close = print})
    getmetatable(xyz).__close = nil
    if true then break end
    local z = 1
  until
    false
end
print(pcall(rep))

local function wh ()
  while true do
    local xyz <close> = setmetatable({}, {__close = print})
    getmetatable(xyz).__close = nil
    if true then break end

  end
end
print(pcall(wh))
