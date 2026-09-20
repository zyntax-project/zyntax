-- integer loops at the edges of the range: the step never wraps past the limit
local function checkfor(from, to, step)
  local out = {}
  for i = from, to, step do out[#out + 1] = i end
  print(table.concat(out, " "))
end
local maxi, mini = math.maxinteger, math.mininteger
checkfor(mini, maxi, maxi)
checkfor(mini, math.huge, maxi)
checkfor(maxi, mini, mini)
checkfor(maxi, mini, -maxi)
checkfor(maxi, -math.huge, mini)
checkfor(maxi, mini, 1)
checkfor(mini, maxi, -1)
checkfor(maxi - 2, maxi, 1)
checkfor(mini + 2, mini, -1)
checkfor(1, 10, 4)
checkfor(10, 1, -4)
checkfor(1, 10, 3)
checkfor(maxi - 5, maxi, 3)
checkfor(mini + 5, mini, -3)
checkfor(maxi, maxi, maxi)
checkfor(maxi, maxi, mini)
checkfor(mini, mini, maxi)
checkfor(mini, mini, mini)
local n = 5
for i = 1, n do io.write(i, " ") end print()
for i = n, 1, -1 do io.write(i, " ") end print()
for i = 1, n, 2 do io.write(i, " ") end print()
for i = 1, 3 do io.write(i, " ") end print()
for i = 3, 1, -1 do io.write(i, " ") end print()
for i = 1, 10, 4 do io.write(i, " ") end print()
-- a float limit is the last integer the loop reaches, in the step's direction
local c
c = 0; for i = 1, 0.99999, -1 do c = c + 1 end; print(c)
c = 0; for i = 1, 0.99999, 1 do c = c + 1 end; print(c)
c = 0; for i = 9999, 1e4, -1 do c = c + 1 end; print(c)
c = 0; for i = 1, 10.9 do c = c + 1 end; print(c)
c = 0; for i = 10, 0.001, -1 do c = c + 1 end; print(c)
c = 0; for i = 1, "10.8" do c = c + 1 end; print(c)
c = 0; for i = 9, "3.4", -1 do c = c + 1 end; print(c)
c = 0; for i = 100, "96.3", -2 do c = c + 1 end; print(c)
local lim, st = 2.5, -1
for i = 3, lim, st do io.write(i, " ") end print()
for i = 3, 1.5, -1 do io.write(i, " ") end print()
for i = 1, 2.5 do io.write(i, " ") end print()
for i = 1.0, 3 do io.write(i, " ") end print()
-- a limit past the integers, against the step or with it
for i = mini, -10e100 do print("no") end
for i = maxi, 10e100, -1 do print("no") end
c = 0; for i = 1, math.huge do if i > 10 then break end; c = c + 1 end; print(c)
c = 0; for i = -1, -math.huge, -1 do if i < -10 then break end; c = c + 1 end; print(c)
local huge = 10e100
for i = mini, -huge do print("no") end
for i = maxi - 1, huge do io.write(i, " ") end print()
-- a zero step is an error, literal or not
print(pcall(function () for i = 1, 10, 0 do end end))
print(pcall(function () for i = 1.0, -10, 0.0 do end end))
local z = 0
print(pcall(function () for i = 1, 10, z do end end))
