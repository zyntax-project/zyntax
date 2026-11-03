local Base = {}
Base.__index = Base
function Base.new(value)
  return setmetatable({value = value}, Base)
end
local Full = setmetatable({}, {__index = Base})
Full.__index = Full
Full.__newindex = function(o, key, value)
  if key ~= "count" then rawset(o, key, value) end
end
function Full.new(value, limit)
  local o = Base.new(value)
  setmetatable(o, Full)
  o.limit = limit
  o.count = 0
  return o
end
function Full:advance()
  self.count = self.count + 1
  if self.count >= self.limit then
    self.value = not self.value
    self.count = 0
  end
  return self.count
end
local t = Full.new(true, 3)
print(t.count == nil)
local ok, err = pcall(function() return t:advance() end)
print(ok, type(err))
