-- string.dump(f, true): a function without debug information.

local function f(a)
  local b = a + 1
  return b
end
local s = load(string.dump(f, true))
print(s(1))

local t = debug.getinfo(s)
print(t.source, t.short_src, t.what, t.linedefined, t.lastlinedefined)
print(t.currentline, next(debug.getinfo(s, "L").activelines))

-- runtime errors are at '?:-1:' with no variable names
local function add(a) return a + 1 end
local sa = load(string.dump(add, true))
print(pcall(sa, {}))
local sb = load(string.dump(function () local q; q = {}; return q + 2 end, true))
print(pcall(sb))
print(pcall(load(string.dump(function () undefinedname() end, true))))
print(pcall(load(string.dump(function () local z = {}; return z.x.y end, true))))

-- error() in stripped code has no position
print(pcall(load(string.dump(function () error("boom") end, true))))

-- upvalue names are '(no name)'; setupvalue still works
local up = 12
local ups = load(string.dump(function () return up end, true))
local un, uv = debug.getupvalue(ups, 1)
print(un, uv == _ENV)
print(debug.setupvalue(ups, 1, 99), ups())

-- a stripped chunk: locals are temporaries, lines are -1
local prog = [[
local a = 12
local n, v = debug.getlocal(1, 1)
print(n, v)
print(debug.getinfo(1).currentline)
local g = function () local x; return a end
local gn, gv = debug.getupvalue(g, 1)
print(gn, gv)
print(debug.setupvalue(g, 1, 13), a)
local i = debug.getinfo(g)
print(i.short_src, i.linedefined > 0, i.lastlinedefined == i.linedefined)
print(next(debug.getinfo(g, "L").activelines))
local g2 = load(string.dump(g))
i = debug.getinfo(g2)
print(i.short_src, i.linedefined > 0, i.lastlinedefined == i.linedefined)
return a
]]
local p = assert(load(string.dump(load(prog), true)))
print(p())

-- line hooks in stripped code get no line
local function foo()
  local a = 1
  local b = 2
  return b
end
local sf = load(string.dump(foo, true))
local line = true
debug.sethook(function (e, l) line = l end, "l")
local r = sf(); debug.sethook(nil)
print(r, line)

-- sizes: the source name is stored once, and not at all when stripped
local body = [[
  return function (x)
    return function (y)
      return x + y
    end
  end
]]
local name = string.rep("x", 1000)
local pn = assert(load(body, name))
local c = string.dump(pn)
print(#c > 1000 and #c < 2000)
local lf = assert(load(c))
print(lf()(3)(5))
print(debug.getinfo(lf).source == name, debug.getinfo(lf()).source == name)
local cs = string.dump(pn, true)
print(#cs < 500)
local ls = assert(load(cs))
print(ls()(30)(50), debug.getinfo(ls).source, debug.getinfo(ls()(1)).source)
