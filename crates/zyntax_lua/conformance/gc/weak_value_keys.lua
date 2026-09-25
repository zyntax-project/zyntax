-- The key of a weak-value entry whose value dies: the collection that
-- clears the value keeps the key, and the next one takes it when
-- nothing else holds it. Only the program's own collections run, so
-- which collection clears what does not depend on when the reference
-- would have collected by itself.
collectgarbage("stop")
local function scrub(n)
  local a, b, c, d, e = n, n, n, n, n
  if n > 0 then return scrub(n - 1) + a end
  return 0
end
local function count(t)
  local n = 0
  for _ in pairs(t) do n = n + 1 end
  return n
end

local vt = setmetatable({}, {__mode = "v"})
local watch = setmetatable({}, {__mode = "k"})
local holder = {}
local function fill()
  holder.held = {}
  for i = 1, 5 do
    local key = {id = i}
    vt[key] = {i}
    watch[key] = true
  end
  for i = 6, 10 do
    local key = {id = i}
    local value = {i}
    vt[key] = value
    watch[key] = true
    holder.held[i] = value
  end
end
fill()
scrub(200)
print("before", count(vt), count(watch))
collectgarbage()
print("after 1", count(vt), count(watch))
holder.held = nil
scrub(200)
collectgarbage()
print("after 2", count(vt), count(watch))
scrub(200)
collectgarbage()
print("after 3", count(vt), count(watch))

-- A key with a finalizer whose value died is finalized while the
-- program runs, not at its end.
local mt = {__gc = function(o) print("finalized key", o.id) end}
local vg = setmetatable({}, {__mode = "v"})
local function fill_gc()
  for i = 1, 3 do vg[setmetatable({id = i}, mt)] = {} end
end
fill_gc()
for _ = 1, 3 do
  scrub(200)
  collectgarbage()
end
print("keys collected", count(vg))

-- A string key keeps no entry once its value is gone, and a key whose
-- value lives on stays.
local vs = setmetatable({}, {__mode = "v"})
local keep = {}
local function fill_str(r)
  for i = 1, 200 do vs["k" .. r .. "_" .. i] = {} end
  keep[r] = {}
  vs["kept" .. r] = keep[r]
end
for r = 1, 5 do
  fill_str(r)
  scrub(200)
  collectgarbage()
end
print("string keys", count(vs), vs.kept1 == keep[1], vs.kept5 == keep[5])
print("end of program")
