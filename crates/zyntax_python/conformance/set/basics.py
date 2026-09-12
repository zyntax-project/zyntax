# test_set: literal, add, in, len, ops
s = {1, 2, 3}
s.add(2)
s.add(4)
print(len(s))
print(2 in s, 9 in s)
print(sorted(s))
t = {3, 4, 5}
print(sorted(s & t))
print(sorted(s | t))
print(sorted(s - t))
print(sorted({x % 3 for x in range(10)}))
