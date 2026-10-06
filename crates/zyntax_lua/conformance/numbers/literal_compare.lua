local function probe(x)
  print(x == 1, 1 == x, x ~= 1, 1 ~= x,
        x < 1, 1 < x, x <= 1, 1 <= x,
        x > 1, 1 > x, x >= 1, 1 >= x)
  print(x == 0, x < 0, 0 < x, x <= 0, 0 <= x,
        x == 9007199254740992, x < 9007199254740992,
        9007199254740992 < x, x <= 9007199254740992,
        9007199254740992 <= x)
  print(x == 9007199254740993, 9007199254740993 == x,
        x < 9007199254740993, 9007199254740993 < x,
        x <= 9007199254740993, 9007199254740993 <= x)
end

probe(1)
probe(1.0)
probe(1.5)
probe(-0.0)
probe(0 / 0)
probe(math.huge)
probe(-math.huge)
probe(9007199254740992.0)
probe(9007199254740993)
probe(9007199254740994.0)
probe(-9007199254740992.0)

local calls = 0
local function value(x)
  calls = calls + 1
  return x
end
print(value(1.5) < 2, 2 >= value(1.5), value(0 / 0) ~= 1,
      1 == value(1.0), value(-0.0) <= 0, 0 < value(math.huge))
print(calls)
