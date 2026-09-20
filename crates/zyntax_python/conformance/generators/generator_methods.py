class Grid(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h

    def indexes(self):
        for y in range(self.height):
            for x in range(self.width):
                yield x, y

    def evens(self, limit):
        for i in range(limit):
            if i % 2 == 0:
                yield i


def main():
    g = Grid(3, 2)
    print(list(g.indexes()))
    total = 0
    for x, y in g.indexes():
        total += x * y
    print(total, sum(g.evens(7)))
    grids = [Grid(1, 1), Grid(2, 1)]
    for grid in grids:
        print(list(grid.indexes()))


main()
