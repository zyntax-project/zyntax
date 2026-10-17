# A function that widens a module-level set or dict, by changing it in
# place or by passing it to a parameter that takes a wider one, without
# declaring it global.
ints = set([1, 2])
table = {1: 2}
rows = [[(1, 2)], [(3, 4)]]
groups = set([frozenset(x) for x in rows])


def widen():
    ints.add("x")
    table["k"] = 2.5
    return len(ints) + len(table)


def picked():
    return list(filter(lambda r: True, rows))


def remove(r, n):
    if n == 0:
        return len(r)
    taken = frozenset(picked()[0])
    return remove(r - taken, n - 1)


def run():
    return remove(groups, 2)


print(widen(), len(ints), len(table), "x" in ints, table["k"])
print(run(), len(groups))
