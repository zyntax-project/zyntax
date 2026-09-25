-- While a message handler runs for a stack overflow, another overflow
-- is an error in error handling, a message with no position.
local function loop (x, y, z) return 1 + loop(x, y, z) end

local res, msg = xpcall(loop, function (m)
  print(string.find(m, "stack overflow") ~= nil)
  print(pcall(loop))
  print(math.sin(0))
  return 15
end)
print(res, msg)

-- An overflow in the handler that nothing catches ends the xpcall.
print(xpcall(loop, function (m) return loop() end))

-- Outside a handler, an overflow is an overflow again.
local ok, err = pcall(loop)
print(ok, string.find(err, "stack overflow") ~= nil)

-- A handler for another error sees ordinary overflows.
print(xpcall(error, function (m)
  local ok, err = pcall(loop)
  return (string.find(err, "stack overflow") ~= nil)
end, "boom"))
