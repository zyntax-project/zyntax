local function deep(n) return deep(n + 1) + 1 end
local ok, err = pcall(deep, 1)
print(ok, string.find(err, "stack overflow") ~= nil)
local function count(n) if n == 0 then return 0 end return 1 + count(n - 1) end
print(count(10000))
local depth = 0
local function probe() depth = depth + 1; probe() end
print(pcall(probe))
print(depth > 1000)
local co = coroutine.wrap(function() local function d() return d() + 1 end return pcall(d) end)
local ok2, err2 = co()
print(ok2, string.find(tostring(err2), "stack overflow") ~= nil)
print(select("#", pcall(deep, 1)))
print("after")
