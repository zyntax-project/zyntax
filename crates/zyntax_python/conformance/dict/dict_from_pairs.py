# dict() of a generator or list of pairs is a dict of those keys and
# values, read back, tested for membership and iterated.
board = [10, 20, 30, 40]
cti = dict((board[i], i) for i in range(len(board)))
print(cti[20], cti[40], 30 in cti, 35 in cti, cti.get(35, -1))
names = dict([(str(i), i * i) for i in range(4)])
print(names["3"], sorted(names), sum(names.values()))
pairs = [("x", 1.5), ("y", 2.5)]
byname = dict(pairs)
print(byname["y"] + byname["x"])
total = 0
for k in cti:
    total += cti[k] + k
print(total)
flipped = dict((v, k) for k, v in cti.items())
print(flipped[2], len(flipped))
empty = dict((k, k) for k in [])
print(empty, len(empty))
