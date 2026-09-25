-- debug.getupvalue, setupvalue, upvalueid and upvaluejoin.
local a, b = 1, "two"
local function f()
  return a, b
end
print(debug.getupvalue(f, 1))
print(debug.getupvalue(f, 2))
print(debug.getupvalue(f, 3))

local function g()
  print(a)
  return b
end
print(debug.getupvalue(g, 1) , select(2, debug.getupvalue(g, 1)) == _G)
print(debug.getupvalue(g, 2))
print(debug.getupvalue(g, 3))

local function counter()
  local n = 0
  return function()
    n = n + 1
    return n
  end
end
local c1, c2 = counter(), counter()
c1(); c1()
print(debug.getupvalue(c1, 1))
print(debug.getupvalue(c2, 1))
print(debug.setupvalue(c1, 1, 10))
print(c1())
print(debug.setupvalue(c1, 5, 10))
print(debug.upvalueid(c1, 1) == debug.upvalueid(c1, 1))
print(debug.upvalueid(c1, 1) == debug.upvalueid(c2, 1))
print(type(debug.upvalueid(c1, 1)))

debug.upvaluejoin(c1, 1, c2, 1)
print(c1(), c2())
print(debug.upvalueid(c1, 1) == debug.upvalueid(c2, 1))
local ok, err = pcall(debug.upvaluejoin, c1, 3, c2, 1)
print(ok, err:match("%((.-)%)$"))
ok, err = pcall(debug.getupvalue, 1, 1)
print(ok, err:match("%((.-)%)$"))
print(debug.getupvalue(print, 1))

local function env_user() return tostring end
local name, env = debug.getupvalue(env_user, 1)
print(name, env == _G)
