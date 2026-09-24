-- A memo cache under weak values shrinks once its results are no
-- longer used, and keeps working afterwards.
local cache = setmetatable({}, {__mode = "v"})
local made = 0
local function get(n)
  local r = cache[n]
  if r == nil then
    made = made + 1
    r = {value = n * n}
    cache[n] = r
  end
  return r
end
local function count()
  local c = 0
  for _ in pairs(cache) do c = c + 1 end
  return c
end
local function use(from, to)
  local sum = 0
  for i = from, to do sum = sum + get(i).value end
  return sum
end
print(use(1, 1000))
print("before", count() <= 1000)
local kept = get(7)
collectgarbage()
collectgarbage()
print("after shrinks", count() < 50)
print("kept", cache[7] == kept, kept.value)
print(use(1, 10), made >= 1000)
