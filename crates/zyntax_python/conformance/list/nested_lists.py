# Lists of lists hold the inner lists themselves: an alias, a copy of
# the outer list and a list built from it all see one inner list.
from array import array


class Done:
    def __init__(self, count):
        self.cells = [[0, 1] for i in range(count)]

    def remove(self, i, v):
        self.cells[i].remove(v)

    def set_done(self, i, v):
        self.cells[i] = [v]


def extend_through(o):
    o[0].append(99)
    o.append([5])


def aliases():
    xs = [[1, 2], [3]]
    a = xs[0]
    a.append(9)
    print(xs, a is xs[0], len(xs[0]))
    ys = xs[:]
    ys[1].append(4)
    print(xs, ys, ys[1] is xs[1], ys is xs)
    zs = list(xs)
    zs[0].append(7)
    print(xs[0], zs[0] is xs[0])
    cp = xs.copy()
    cp.append([8])
    print(len(xs), len(cp))
    rep = [[0]] * 3
    rep[0].append(1)
    print(rep)


def fields():
    d = Done(3)
    held = d.cells[1]
    d.set_done(1, 5)
    held.append(6)
    print(d.cells, held)
    d.remove(0, 1)
    print(d.cells)


def through_parameters():
    xs = [[1, 2], [3]]
    extend_through(xs)
    print(xs)
    mixed = [xs, "s"]
    extend_through(mixed[0])
    print(xs)


def three_levels():
    t = [[[1]], [[2, 3]]]
    t[1][0].append(4)
    inner = t[0]
    inner.append([5])
    t[0][1].append(6)
    print(t, len(t[1][0]))


def ordering():
    rows = [[2, 1], [1, 5], [1, 2]]
    print(sorted(rows), max(rows), min(rows))
    rows.sort()
    print(rows, rows.index([1, 5]), [1, 2] in rows, [9] in rows)
    print(rows == [[1, 2], [1, 5], [2, 1]], rows != [[1, 2]])
    words = [["b", "a"], ["a"]]
    words.sort()
    print(words, [["a"]] < [["b"]])


def arrays():
    rows = [array('d', [0.0]) * 3 for y in range(2)]
    rows[1][2] = 2.5
    first = rows[0]
    first[0] = 1.5
    print(rows, len(rows[0]), rows[1][2], rows[0] is first)
    for r in rows:
        r[1] = 7.0
    print(rows)


def printing():
    grid = [[0] * 3 for i in range(2)]
    grid[1][2] = 5
    print(grid, str(grid), repr(grid), f"{grid}")
    print([[1.5], [2.5, 3.5]], [["x"], []], [[(1, "a")], [(2, "b")]])
    print(sum(len(r) for r in grid), [v for r in grid for v in r])


aliases()
fields()
through_parameters()
three_levels()
ordering()
arrays()
printing()
