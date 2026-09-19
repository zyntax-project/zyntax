# the array module: an array is stored at its typecode's width
from array import array
import array as arr

# construction: empty, from a list of the kind, converted from ints,
# from a tuple, a range, a generator and another array; copies share nothing
a = array('d')
a.append(1.5)
print(a, len(a))
b = array('d', [0])
print(b, b[0] * 2)
c = array('i', (1, 2, 3))
print(c, sum(c), c[-1])
d = array('d', range(4))
print(d)
e = arr.array('q', (x * x for x in range(5)))
print(e)
src = [1, 2, 3]
f = array('l', src)
f.append(4)
print(src, f)
g = array('d', b)
g[0] = 9.0
print(b[0], g[0])
print(array('i'), array('B', [65, 66]), array('h', array('b', [-1, 2])))

# repeat, concatenation, slicing, slice assignment, equality, order
z = array('d', [0]) * 5
print(z, len(z))
z[2] = 2.5
print(z[1:4], z + array('d', [7.0]))
z[:] = array('d', [1.0]) * 5
print(z)
print(array('i', [1, 2]) == array('i', [1, 2]), array('i', [1, 2]) == array('i', [2, 1]))
print(array('i', [1, 2]) == array('l', [1, 2]), array('i', [1, 2]) == [1, 2], array('i', [1]) != [1])
print(array('b', [1, 2]) < array('b', [1, 3]), array('B', [3]) >= array('B', [2, 9]))
print(3 * array('B', [1, 2]), array('f', [0.5]) * 2)

# the list methods an array has
h = array('i', [5, 3, 8])
h.extend([1, 9])
h.fromlist([4])
h.insert(0, 7)
print(h, h.index(8), h.count(1), h.pop(), len(h))
h.remove(3)
h.reverse()
h.extend(array('i', [2, 2]))
print(h, h.count(2), h.tolist())
print(min(h), max(h), sorted(h), list(h), tuple(array('B', [7, 8])))
print(h.typecode, h.itemsize, array('d').itemsize, array('B').itemsize, array('h').itemsize)

# iteration, membership, indexing in loops
total = 0.0
m = array('d', [0.5, 1.5, 2.5])
for v in m:
    total += v
print(total, 1.5 in m, 4.0 in m)
n = 6
data = array('d', [0]) * (n * n)
for y in range(n):
    for x in range(n):
        data[y * n + x] = x * 0.5 + y
acc = 0.0
for i in range(n * n):
    acc += data[i]
print(acc, data[7], data[35])
for i, v in enumerate(array('B', [9, 8])):
    print(i, v)

# every integer code reads back as an int; a value out of the code's range
# is an OverflowError, a value that could never be stored is in no array
bs = array('b', [-128, 127])
print(bs, bs[0] - 1, bs[1] + 1, 200 in bs, bs.count(-129))
us = array('B', [0, 255])
us[0] = 254
print(us, us[0] + us[1])
hs = array('h', [-32768, 32767])
print(hs, sum(hs))
uh = array('H', [65535])
print(uh, uh[0] * 2)
ii = array('i', [-2147483648, 2147483647])
print(ii, ii[1] + 1)
ui = array('I', [4294967295])
print(ui, ui[0] + 1)
ls = array('l', [-9223372036854775807 - 1, 9223372036854775807])
print(ls, ls[1])
qs = array('Q', [9223372036854775807])
print(qs[0])
for code, bad in (('b', 128), ('b', -129), ('b', 40000), ('B', -1), ('B', 256), ('h', 32768),
                  ('H', 70000), ('H', -1), ('i', 2147483648), ('I', -1), ('I', 4294967296),
                  ('L', -1)):
    try:
        if code == 'b':
            array('b', [0]).append(bad)
        elif code == 'B':
            array('B', [0])[0] = bad
        elif code == 'h':
            array('h', [bad])
        elif code == 'H':
            array('H').insert(0, bad)
        elif code == 'i':
            array('i').extend([bad])
        elif code == 'I':
            array('I').fromlist([bad])
        else:
            array('L', [1]).append(bad)
    except OverflowError as e:
        print(code, bad, 'OverflowError:', e)
print(arr.typecodes)

# a float array converts ints on the way in; 'f' rounds to 32 bits
w = array('d', [1, 2, 3])
w.append(4)
print(w, w[0] / 2)
fs = array('f', [0.1, 1.5])
print(fs[0], fs[1], fs, fs.tolist(), 0.1 in fs, 1.5 in fs, sum(fs))
fs[0] = 2.0
fs.append(1)
print(fs, array('f', fs) == fs, array('d', fs))

# mixing arrays with lists is a TypeError
try:
    print(array('i', [1]) + [2])
except TypeError as e:
    print('TypeError:', e)
try:
    array('i', [1]).extend(array('l', [2]))
except TypeError as e:
    print('TypeError:', e)
try:
    print(array('i', [1]) < [2])
except TypeError as e:
    print('TypeError:', e)
try:
    array('i').index(7)
except ValueError as e:
    print('ValueError:', e)
try:
    array('B').remove(300)
except ValueError as e:
    print('ValueError:', e)

# an array in a class field, a tuple and a dynamic slot keeps its typecode
class Grid(object):
    def __init__(self, w, h):
        self.width = w
        self.data = array('d', [0]) * (w * h)

    def get(self, x, y):
        return self.data[y * self.width + x]

grid = Grid(3, 2)
grid.data[4] = 6.5
print(grid.get(1, 1), grid.data)
pair = (array('B', [1]), 'x')
first = pair[0]
first.append(2)
print(pair, first)
objs = [array('i', [1]), 'text']
print(objs)
