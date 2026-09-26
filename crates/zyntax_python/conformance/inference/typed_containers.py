# Sets and dicts typed by what goes in them: a set of ints filled by
# adds, a dict keyed by tuples, dicts held in a list and frozensets in a
# list of lists, each read back by position and shared with its holder.
class Board:
    def __init__(self):
        self.seen = set()
        self.cost = {}

    def visit(self, x, y):
        self.seen.add(x * 10 + y)
        self.cost[(x, y)] = x + y


def main():
    s = set()
    for i in range(12):
        s.add(i % 5)
    print(sorted(s), 3 in s, 7 in s, min(s), len(s))
    grid = {}
    for x in range(3):
        for y in range(3):
            grid[(x, y)] = x * y
    print(grid[(2, 2)], (1, 2) in grid, (3, 3) in grid, len(grid))
    rows = []
    for i in range(4):
        rows.append({"id": i, "sq": i * i})
    rows[1]["sq"] = 100
    alias = rows[2]
    alias["id"] = 20
    print([r["id"] + r["sq"] for r in rows], rows[2]["id"])
    layers = [[frozenset([1, 2]), frozenset([3])], [frozenset([2, 1])]]
    print(layers[0][0] == layers[1][0], sorted(layers[0][0] | layers[0][1]), len(layers[1]))
    print(sorted(sorted(f) for row in layers for f in row))
    print(sorted([frozenset([3]), frozenset([1, 2, 3]), frozenset([1, 3])], key=len))
    b = Board()
    for x in range(3):
        b.visit(x, x + 1)
    print(sorted(b.seen), b.cost[(1, 2)], sorted(b.cost.values()))
    nums = [{1, 2}, {3}]
    nums[0].add(9)
    print(sorted(nums[0]), nums[1] < nums[0], nums[1] <= {3, 4})


main()
