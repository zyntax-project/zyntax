-- metamethod frames, and the names argument errors give.
local mt = setmetatable({}, {
  __index = function(t, k)
    local i = debug.getinfo(1, "n")
    return i.name .. ":" .. i.namewhat
  end,
  __newindex = function(t, k, v)
    local i = debug.getinfo(1, "n")
    print(i.name, i.namewhat)
  end,
})
print(mt.foo)
mt.bar = 1
local V = setmetatable({}, {__add = function(a, b)
  local i = debug.getinfo(1, "n")
  return i.name .. ":" .. i.namewhat
end})
print(V + 1)

print(pcall(debug.traceback, "x", "notnum"))
print(pcall(debug.getinfo, "x"))
print(pcall(debug.getinfo, 1, 2))
print(pcall(debug.getinfo, 1, ">"))
print(pcall(debug.getinfo, {}))
print(pcall(debug.getinfo))
print(pcall(debug.getupvalue, 1, 1))
print(pcall(debug.setupvalue, print, 1))
print(pcall(debug.upvaluejoin, function() return print end, 1, print, 1))
print(pcall(debug.getlocal, 50, 1))
print(pcall(debug.getlocal, 1))
print(pcall(debug.setlocal, 1, 1))
print(pcall(debug.sethook, print, "c", "x"))
print(debug.gethook())
print(pcall(function() return debug.getinfo("x") end))
