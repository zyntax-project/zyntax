local sum = 0
for i = 1, 1000000 do
  sum = (sum + i) % 997
end
print(sum)
