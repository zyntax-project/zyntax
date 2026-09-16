def compare(a, b):
    print(a <= b, a < b, b >= a, b > a)

small = frozenset([0, 1, 2, 3, 8])
large = frozenset(range(50))
compare(small, large)
compare(frozenset([1, 7]), frozenset([1, 4]))
compare(frozenset([0, 62]), frozenset([0, 62]))
compare(frozenset([0, 63]), frozenset([0, 62]))
compare(frozenset([1.0]), frozenset([1]))
print(sorted(frozenset([0, 62]) & frozenset([62, 63])))
print(sorted(frozenset([0, 62]) - frozenset([62, 63])))
print(len(frozenset([1.0, 2.0]) & frozenset([1, 3])))
rows = set([frozenset([(0, 0), (0, 1)]), frozenset([(1, 0)])])
positions = set([(0, 0), (0, 1)])
print(len(rows - positions), len(rows & positions), rows <= positions)
print(len(positions - rows), len(positions & rows), positions <= rows)

def minimum(values):
    return min(values)

print(minimum(large - small))
