# A list field changed through a receiver of no known class may be
# given anything, so it keeps whatever it is given.
class Grid:
    def __init__(self):
        self.cells = [[1, 2], [3]]


class Other:
    def __init__(self):
        self.cells = 0


def poke(o):
    o.cells[0] = "u"


def grow(o):
    o.cells.append("w")


def main():
    g = Grid()
    poke([g, Other()][0])
    print(g.cells, len(g.cells[0]))
    grow([g, Other()][0])
    print(g.cells, len(g.cells))


main()
g = Grid()
poke([g, Other()][0])
print(g.cells, len(g.cells[0]))
