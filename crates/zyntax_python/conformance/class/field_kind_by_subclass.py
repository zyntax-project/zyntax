# A base class's field holds one array; a subclass stores a list of
# arrays in the same field. The base's methods store numbers through it
# and the subclass's store rows, each on its own instances. The base is
# only made through a class value, so its own write is typed late.
from array import array


class Grid(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.data = array('d', [0]) * (w * h)

    def __getitem__(self, xy):
        x, y = xy
        return self.data[y * self.width + x]

    def __setitem__(self, xy, val):
        x, y = xy
        self.data[y * self.width + x] = val


class Rows(Grid):
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.data = [array('d', [0]) * w for y in range(h)]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.data[idx[1]][idx[0]]
        return self.data[idx]

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            self.data[idx[1]][idx[0]] = val
        else:
            self.data[idx] = val


def smooth(g, n):
    for y in range(1, n - 1):
        for x in range(1, n - 1):
            g[x, y] = (g[x - 1, y] + g[x + 1, y]) * 0.5 + 1.0


def build(kind: object, n):
    return kind(n, n)


def main():
    for name, g in (("Grid", build(Grid, 4)), ("Rows", Rows(4, 4))):
        for y in range(4):
            for x in range(4):
                g[x, y] = float(x * y)
        smooth(g, 4)
        print(name, [g[x, 2] for x in range(4)], g[1, 1])
    r = Rows(2, 2)
    r[1] = array('d', [5.0, 6.0])
    print(list(r[1]), r[0, 1])


main()
