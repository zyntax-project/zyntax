-- the table library and pairs go through the metamethods
local function test(proxy, t)
  for i = 1, 10 do
    table.insert(proxy, 1, i)
  end
  print(#proxy, #t, proxy[1])
  for i = 1, 10 do
    assert(t[i] == 11 - i)
  end
  table.sort(proxy)
  for i = 1, 10 do
    assert(t[i] == i and proxy[i] == i)
  end
  print(table.concat(proxy, ","))
  for i = 1, 8 do
    assert(table.remove(proxy, 1) == i)
  end
  print(#proxy, #t)
  print(table.unpack(proxy))
end

-- all virtual
local t = {}
local proxy = setmetatable({}, {
  __len = function () return #t end,
  __index = t,
  __newindex = t,
})
test(proxy, t)

-- only __newindex
local count = 0
t = setmetatable({}, {
  __newindex = function (t, k, v) count = count + 1; rawset(t, k, v) end})
test(t, t)
print(count)

-- __pairs in a for loop, with a plain table alongside
local p = setmetatable({}, {__pairs = function (t)
  local i = 0
  return function () i = i + 1; if i <= 3 then return i, i * i end end, t, nil
end})
for k, v in pairs(p) do print(k, v) end
local plain = {5, 6, a = 1}
for k, v in pairs(plain) do print(k, v) end

-- a yield inside __pairs
do
  local t = setmetatable({10, 20, 30}, {__pairs = function (t)
    local inc = coroutine.yield()
    return function (t, i)
             if i > 1 then return i - inc, t[i - inc] else return nil end
           end, t, #t + 1
  end})
  local res = {}
  local co = coroutine.wrap(function ()
    for i, p in pairs(t) do res[#res + 1] = p end
  end)
  co()
  co(1)
  print(res[1], res[2], res[3], #res)
end

-- table.remove on an empty table takes the entry at 0
local a = {[0] = "ban"}
print(#a, table.remove(a), a[0])
a = {[-1] = "ban"}
print(#a, table.remove(a), a[-1])
