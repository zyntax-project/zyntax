from array import array


class Grid(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.data = array('d', [0]) * (w * h)

    def _idx(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return y * self.width + x
        raise IndexError

    def __getitem__(self, x_y):
        (x, y) = x_y
        return self.data[self._idx(x, y)]

    def __setitem__(self, x_y, val):
        (x, y) = x_y
        self.data[self._idx(x, y)] = val

    def indexes(self):
        for y in range(self.height):
            for x in range(self.width):
                yield x, y


class Rows(Grid):
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.data = [array('d', [0]) * w for y in range(h)]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.data[idx[1]][idx[0]]
        else:
            return self.data[idx]

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            self.data[idx[1]][idx[0]] = val
        else:
            self.data[idx] = val

    def copy_data_from(self, other):
        for l1, l2 in zip(self.data, other.data):
            l1[:] = l2


def fill(g):
    for x, y in g.indexes():
        g[x, y] = x + 10 * y


def main():
    g = Grid(3, 2)
    fill(g)
    print(g[2, 1], list(g.data))
    r = Rows(3, 2)
    fill(r)
    print(r[2, 1], r[1], [list(row) for row in r.data])
    r[0], r[1] = r[1], r[0]
    print([list(row) for row in r.data])
    r[1][2] *= 2.0
    r[0][0] -= r[1][2] * 0.5
    print([list(row) for row in r.data])
    other = Rows(3, 2)
    other.copy_data_from(r)
    print([list(row) for row in other.data])
    boxed = [g, r]
    for b in boxed:
        print(b[1, 1])
        b[1, 1] = 99
    print(g[1, 1], r[1, 1])
    try:
        g[5, 5]
    except IndexError:
        print("IndexError")


main()
