-- os.exit(code, true) closes the main chunk's variables first.
local x <close> = setmetatable({}, {__close = function (_, err)
  assert(err == nil)
  print("Ok")
end})
local e1 <close> = setmetatable({}, {__close = function () print(120) end})
local function inner ()
  local y <close> = setmetatable({}, {__close = function () print("inner") end})
  print(pcall(os.exit, true, true))
  print("not reached")
end
inner()
print("not reached")
