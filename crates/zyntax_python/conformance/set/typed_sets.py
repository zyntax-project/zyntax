f = frozenset([3, 1, 2])
print(len(f), 2 in f, 5 in f, 2.0 in f, True in frozenset([1]))
print(repr(frozenset()), frozenset([7]))
try:
    f.add(4)
except AttributeError as e:
    print("AttributeError", e)
s = {1, 2}
s.add(3)
print(sorted(s), sorted(s | f), sorted(f - s), f <= s, s >= f, f < s, s == f)
g = f | {9}
print(sorted(g), isinstance(g, frozenset), isinstance(s | f, set))
print(min(f), max(s))
e = set()
e.add(2.5)
e.add(1.5)
print(sorted(e))
u = frozenset.union(f, {10}, [11])
print(sorted(u))
t = set((i, i + 1) for i in range(3))
print(sorted(t), (1, 2) in t, (1, 3) in t)
w = {x * 2 for x in range(5) if x % 2}
w.discard(2)
w.discard(7)
w.remove(6)
print(sorted(w), sorted(w.union([1.0, 2])), w.isdisjoint({1}), w.issubset(range(10)))
