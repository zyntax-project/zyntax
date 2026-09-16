def compare(a, b):
    print(a <= b, a < b, b >= a, b > a)

small = frozenset([0, 1, 2, 3, 8])
large = frozenset(range(50))
compare(small, large)
compare(frozenset([1, 7]), frozenset([1, 4]))

def minimum(values):
    return min(values)

print(minimum(large - small))
