# A set iterated by a for statement, a comprehension, a generator and
# map, before and after deletions.
def run2():
    s = set([3, 1, 2])
    total = 0
    for x in s:
        total += x
    return total
print(run2())
s = set()
s.add(1)
print(sorted([str(p) for p in s]))
t = set()
for i in range(10):
    t.add(i)
u = set(x for x in t if x % 3 != 0)
print(sorted(u), sorted(map(lambda v: -v, {3, 1, 2})))
big = set(range(100))
for i in range(0, 100, 2):
    big.discard(i)
n = 0
for x in big:
    n += x
print(n, len(big), 51 in big, 50 in big)
