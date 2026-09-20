-- functions, builtins, threads, files and tables as keys
local t = {}
local function f() end
local function g() end
t[f] = 1
t[g] = 2
t[print] = 3
t[string.rep] = 4
t[coroutine.running()] = 5
t[io.stdin] = 6
local k = {}
t[k] = 7
print(t[f], t[g], t[print], t[string.rep], t[coroutine.running()], t[io.stdin], t[k])
print(t[string.len], t[function() end], t[{}])
local n = 0
for key, v in pairs(t) do n = n + 1; assert(t[key] == v) end
print(n)
t[f] = nil
t[print] = nil
print(t[f], t[print], t[g])
-- the same function reached two ways is one key
local p = print
t[p] = "p"
print(t[print], p == print, string.rep == string.rep, f == f, f == g)
-- a set of functions
local seen = {}
for _, fn in ipairs({f, g, f, print, g, print, string.rep}) do
  seen[fn] = (seen[fn] or 0) + 1
end
print(seen[f], seen[g], seen[print], seen[string.rep])
