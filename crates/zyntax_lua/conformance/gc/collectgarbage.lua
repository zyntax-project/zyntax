-- What each collectgarbage option returns.
print(collectgarbage("isrunning"))
print(collectgarbage())
print(collectgarbage("collect"))
print(math.type(collectgarbage("count")), collectgarbage("count") > 0)
print(type(collectgarbage("step")), type(collectgarbage("step", 100)))
print(collectgarbage("incremental"))
print(collectgarbage("generational"), collectgarbage("generational"))
print(collectgarbage("incremental"), collectgarbage("incremental"))
print(collectgarbage("setpause", 100), collectgarbage("setpause"), collectgarbage("setpause", 200))
print(collectgarbage("setstepmul", 50), collectgarbage("setstepmul"), collectgarbage("setstepmul", 100))
print(collectgarbage("stop"), collectgarbage("isrunning"))
local t = {}
for i = 1, 1000 do t[i] = {} end
print(collectgarbage("restart"), collectgarbage("isrunning"))
print(pcall(collectgarbage, "bogus"))
print(pcall(collectgarbage, "setpause", "x"))
local before = collectgarbage("count")
t = nil
collectgarbage()
print(collectgarbage("count") <= before)
