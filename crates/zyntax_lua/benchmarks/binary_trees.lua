local Tree = {}
Tree.__index = Tree

function Tree.new(item, depth)
  local self = setmetatable({}, Tree)
  self.item = item
  if depth > 0 then
    local item2 = item + item
    depth = depth - 1
    self.left = Tree.new(item2 - 1, depth)
    self.right = Tree.new(item2, depth)
  end
  return self
end

function Tree:check()
  if not self.left then return self.item end
  return self.item + self.left:check() - self.right:check()
end

local start = os.clock()

local min_depth = 4
local max_depth = 14
local stretch_depth = max_depth + 1

print(string.format("stretch tree of depth %d check: %d", stretch_depth, Tree.new(0, stretch_depth):check()))

local long_lived_tree = Tree.new(0, max_depth)

local iterations = 1
local d = 0
while d < max_depth do
  iterations = iterations * 2
  d = d + 1
end

local depth = min_depth
while depth <= max_depth do
  local check = 0
  local i = 1
  while i <= iterations do
    check = check + Tree.new(i, depth):check() + Tree.new(-i, depth):check()
    i = i + 1
  end
  print(string.format("%d trees of depth %d check: %d", iterations * 2, depth, check))
  iterations = math.floor(iterations / 4)
  depth = depth + 2
end

print(string.format("long lived tree of depth %d check: %d", max_depth, long_lived_tree:check()))

local elapsed = os.clock() - start
print(string.format("elapsed: %s", elapsed))
