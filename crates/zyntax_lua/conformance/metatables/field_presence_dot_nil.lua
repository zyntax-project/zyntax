local Base = {}
Base.__index = Base
function Base.new(value)
  return setmetatable({value = value}, Base)
end
local Full = setmetatable({}, {__index = Base})
Full.__index = Full
function Full.new(value, limit)
  local o = Base.new(value)
  setmetatable(o, Full)
  o.limit = limit
  o.count = 0
  return o
end
function Full:advance()
  if self == nil then return -2 end
  self.count = self.count + 1
  if self.count >= self.limit then
    self.value = not self.value
    self.count = 0
  end
  return self.count
end
local t = Full.new(true, 3)
print(t:advance(), Full.advance(nil))
