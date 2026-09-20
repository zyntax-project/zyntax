-- an explicit nil environment is not the same as none
local f = load("return _ENV", "=chunk", "t", nil)
print(f())
print(pcall(load("return x", "=chunk", "t", nil)))
print(load("return _ENV == _G", "=chunk")())
print(load("return _ENV == _G", "=chunk", "t")())
local env = {y = 7}
print(load("return y, _ENV == y", "=chunk", "t", env)())
print(load("y = 8; return y", "=chunk", "t", env)(), env.y)
local name = os.tmpname()
local h = io.open(name, "w")
h:write("return _ENV\n")
h:close()
print(loadfile(name, "t", nil)())
print(loadfile(name)() == _G)
print(loadfile(name, "t", env)() == env)
os.remove(name)
-- through a value
local l = load
print(l("return _ENV", "=v", "t", nil)())
print(l("return _ENV", "=v")() == _G)
print(select("#", l("return _ENV", "=v", "t", nil)()))
