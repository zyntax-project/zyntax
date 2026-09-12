# test_tuple: literal, index, unpacking, immutability of shape
t = (1, 2, 3)
print(t)
print(t[0], t[-1])
print(len(t))
a, b, c = t
print(a, b, c)
a, b = b, a
print(a, b)
print(())
print((1,))
print((1, "x", 2.5))
print(t + (4, 5))
