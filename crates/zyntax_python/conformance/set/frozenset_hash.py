# Frozensets hash by their values whatever order they were built in, so
# a set of them finds one by an equal frozenset; each kind prints as
# Python prints it.
fs = set()
fs.add(frozenset([1, 2, 3]))
fs.add(frozenset([3, 2, 1]))
fs.add(frozenset([(0, 1), (1, 0)]))
fs.add(frozenset([1.0, 2.0, 3.0]))
print(len(fs), frozenset([2, 1, 3]) in fs, frozenset([(1, 0), (0, 1)]) in fs)
print(frozenset([1, 2]) in fs, frozenset() in fs)
d = {frozenset([1, 2]): "a"}
print(d[frozenset([2, 1])], frozenset([2.0, 1.0]) in d)
print(len({frozenset([1, 2]), frozenset([2, 1]), frozenset([1.0, 2.0])}))
print({1}, {1.5}, {"a"}, {(1, 2)}, {None}, {True}, set())
print(sorted([len(x) for x in list(fs)]))
big = set()
for i in range(40):
    big.add(frozenset([i, i + 1]))
print(len(big), frozenset([6, 5]) in big, frozenset([5, 7]) in big)
