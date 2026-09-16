a = frozenset([1, 2, 2])
b = frozenset([2, 1])
print(len(a), a == b, 2 in a)
print(len(set([a, b])))

items = list(map(frozenset, [[1, 2], [3]]))
print(len(items), len(items[0]), len(items[1]))

c = frozenset.union(a, frozenset([2, 3]), frozenset([4]))
print(len(c), 1 in c, 3 in c, 4 in c)

def set_ops(left: object, right: object):
    return left - right, left & right, left | right, left ^ right

parts = set_ops(a, frozenset([2, 3]))
print([len(part) for part in parts])
