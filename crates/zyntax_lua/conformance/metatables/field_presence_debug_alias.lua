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
  local name, alias = debug.getlocal(1, 1)
  alias.count = nil
  return self.count == nil
end
local t = Full.new(true, 3)
print(t:advance())
