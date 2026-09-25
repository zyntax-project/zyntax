-- A frame stopped at a call is at the line the call starts on, however
-- many lines its arguments take.
print(debug.traceback("a",
  1))
print(xpcall(function()
  error("x")
end, debug.traceback))
print(xpcall(function()
  local y = 1
end, debug.traceback), debug.traceback("b"
))
local function where()
  return debug.getinfo(2, "l").currentline
end
local t = {
  where(),
  (function() return 1 end)(), where(
  ),
}
print(t[1], t[2], t[3])
local v =
  where()
print(v)
