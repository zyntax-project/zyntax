# Lists that hold ints, floats and None together, and a local that holds
# a list of ints and then a list of mixed numbers.
xs = [1, 2.5, 3]
print(xs, len(xs), xs[0], xs[1], xs[-1], repr(xs[2]))
xs.append(4)
xs.append(0.5)
print(xs, sum(xs), min(xs), max(xs))
print(sorted(xs), sorted(xs, reverse=True))
ys = [x * 2 for x in xs]
print(ys, ys == [2, 5.0, 6, 8, 1.0], 5.0 in ys, 7 in ys, ys.index(6), ys.count(2))
ys[1] = 7
print(ys, ys[1:3], ys[::-1])
zs = [0] * 3 + [1.5]
print(zs, zs + xs)
total = 0
for v in xs:
    total += v
print(total)
def halves(n):
    return [i / 2 if i % 2 else i // 2 for i in range(n)]
h = halves(6)
print(h, [type(v) is int for v in h])
ws = [None, 1, 2.0]
print(ws, ws[0] is None)
ns = list(xs)
ns.sort()
print(ns, ns.pop(), ns.pop(0), ns)
print(str(xs), "%s" % xs)
d = {}
d["k"] = xs
print(d)
print([round(v) for v in [1.4, 2, 2.6]])
def acc(us):
    s = 0
    for j, u in enumerate(us):
        s += u * j
    return s
print(acc([1, 2.0, 3]), acc([1, 2]))


def widened(n):
    u = [1] * n
    for _ in range(2):
        u = [x / 2 for x in u] + [3]
    return u


print(widened(3))
