# Unpacking into float locals from lists, tuples and mixed values.
pairs = [([1.0, 2.0, 3.0], [0.5, 0.25, 0.125], 7.0), ([4.0, 5.0, 6.0], [1.0, 2.0, 3.0], 8.0)]
total = 0.0
for ([x, y, z], v, m) in pairs:
    total += x + y + z + m + v[0]
print(total)
q = (1.5, 2.5)
a, b = q
print(a + b)
r = [1, 2]
c, d = r
print(c - d)
def f(p):
    x, y = p
    return x * y
print(f((2.0, 3.0)), f([4.0, 5.0]))
try:
    x, y, z = [1.0, 2.0]
except ValueError as e:
    print("ValueError", e)
try:
    x, y = 3.5
except TypeError as e:
    print("TypeError")
