local Toggle = {}
Toggle.__index = Toggle

function Toggle.new(start_state)
  return setmetatable({ state = start_state }, Toggle)
end

function Toggle:value()
  return self.state
end

function Toggle:activate()
  self.state = not self.state
  return self
end

local NthToggle = setmetatable({}, { __index = Toggle })
NthToggle.__index = NthToggle

function NthToggle.new(start_state, max_counter)
  local o = Toggle.new(start_state)
  setmetatable(o, NthToggle)
  o.count_max = max_counter
  o.count = 0
  return o
end

function NthToggle:activate()
  self.count = self.count + 1
  if self.count >= self.count_max then
    self.state = not self.state
    self.count = 0
  end
  return self
end

local start = os.clock()

local n = 100000
local val = true
local toggle = Toggle.new(val)

for _ = 1, n do
  val = toggle:activate():value()
  val = toggle:activate():value()
  val = toggle:activate():value()
  val = toggle:activate():value()
  val = toggle:activate():value()
  val = toggle:activate():value()
  val = toggle:activate():value()
  val = toggle:activate():value()
  val = toggle:activate():value()
  val = toggle:activate():value()
end

local ntoggle = NthToggle.new(val, 3)

for _ = 1, n do
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
  val = ntoggle:activate():value()
end

local elapsed = os.clock() - start
print(string.format("elapsed: %s", elapsed))
