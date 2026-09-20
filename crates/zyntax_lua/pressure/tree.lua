-- A small tree built, walked and dropped every step. Its nodes are
-- held only by their parents' fields, which the collector reclaims.
local function build(item, depth)
  if depth > 0 then
    return { left = build(2 * item, depth - 1), right = build(2 * item + 1, depth - 1), item = item }
  end
  return { item = item }
end

local function check(t)
  if t.left == nil then return t.item end
  return t.item + check(t.left) + check(t.right)
end

local function main(n)
  local total = 0
  for i = 1, n do
    total = total + check(build(i, 4))
  end
  return total
end

print(main(tonumber(arg[1])))
