-- debug.getinfo of functions and of stack levels.
local function show(t)
  local keys = {}
  for k in pairs(t) do keys[#keys + 1] = k end
  table.sort(keys)
  for _, k in ipairs(keys) do
    local v = t[k]
    if k == "activelines" then
      local ls = {}
      for l in pairs(v) do ls[#ls + 1] = l end
      table.sort(ls)
      v = table.concat(ls, ",")
    elseif k == "func" then
      v = type(v)
    end
    print("", k, v)
  end
end

local function f(a, b, ...)
  local x = a
  return debug.getinfo(1, "nSlutL")
end
print("f at level 1")
show(f())

print("f as a value")
show(debug.getinfo(f, "SLu"))

print("print")
show(debug.getinfo(print))

print("main chunk")
show(debug.getinfo(1, "Sl"))

local function named() return debug.getinfo(1, "n") end
local t = {named = named}
function t.m(self) return debug.getinfo(1, "n") end
local r = named()
print(r.name, r.namewhat)
r = t.named()
print(r.name, r.namewhat)
r = t:m()
print(r.name, r.namewhat)
function glob() return debug.getinfo(1, "n") end
r = glob()
print(r.name, r.namewhat)

local function callee() return debug.getinfo(2, "l").currentline end
print(callee())

local function tailed() return debug.getinfo(1, "nt") end
local function tailer() return tailed() end
r = tailer()
print(r.name, r.namewhat, r.istailcall)

print(debug.getinfo(100))
print(debug.getinfo(0, "n").name)
local ok, err = pcall(debug.getinfo, 1, ">")
print(ok, err:match("%((.-)%)$"))
ok, err = pcall(debug.getinfo, print, "X")
print(ok, err:match("%((.-)%)$"))
print(debug.getinfo(f, "f").func == f)
local function self_ref() return debug.getinfo(1, "f").func end
print(self_ref() == self_ref)
