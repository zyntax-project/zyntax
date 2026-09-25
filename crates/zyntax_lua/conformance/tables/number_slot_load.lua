-- Fields holding an integer or a float, read and written by a chunk
-- compiled at run time as well as by the program.
local Cell = {}
Cell.__index = Cell
function Cell.new(v, w) return setmetatable({ v = v, w = w }, Cell) end
function Cell:sum() return self.v + self.w end
local touch = load([[
  local t = ...
  local before = math.type(t.v) .. "/" .. math.type(t.w)
  t.v = t.v + 1
  t.w = t.w * 0.5
  return before, t.v, t.w
]])
local c = Cell.new(1, 2.0)
print(touch(c))
print(c.v, math.type(c.v), c.w, math.type(c.w), c:sum())
c.v = 3.5
c.w = 4
print(touch(c))
print(c.v, math.type(c.v), c.w, math.type(c.w), c:sum())
local d = Cell.new(2 ^ 53, -0.0)
print(touch(d))
print(d.v, math.type(d.v), 1 / d.w, d:sum())
