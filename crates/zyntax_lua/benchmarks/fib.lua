local function fib(n)
  if n < 2 then return n end
  return fib(n - 1) + fib(n - 2)
end

local start = os.clock()
for _ = 1, 5 do
  fib(28)
end
local elapsed = os.clock() - start
print(string.format("elapsed: %s", elapsed))
