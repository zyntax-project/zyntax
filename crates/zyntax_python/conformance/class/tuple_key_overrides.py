from array import array


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = array('d', range(width * height))

    def _idx(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return y * self.width + x
        raise IndexError

    def __getitem__(self, key):
        x, y = key
        return self.data[self._idx(x, y)]

    def __setitem__(self, key, value):
        x, y = key
        self.data[self._idx(x, y)] = value


class Rows(Grid):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = [array('d', range(width)) for _ in range(height)]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.data[key[1]][key[0]]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.data[key[1]][key[0]] = value
        else:
            self.data[key] = value


class Leaf(Rows):
    pass


def access(grid):
    grid[1, 2] = grid[1, 2] + 0.5
    return grid[1, 2]


base = Grid(4, 3)
rows = Rows(4, 3)
leaf = Leaf(4, 3)
print(access(base), access(rows), access(leaf))
print(list(rows[1]))
boxed = [base, rows, leaf, "other"]
for grid in boxed[:3]:
    print(access(grid))
try:
    print(access(None))
except TypeError:
    print("TypeError")
try:
    print(base[4, 0])
except IndexError:
    print("IndexError")


class Number:
    def __getitem__(self, key):
        return key[0] + key[1]


class Text(Number):
    def __getitem__(self, key):
        return "text"


def different_result(value):
    return value[1, 2]


print(different_result(Number()), different_result(Text()))
