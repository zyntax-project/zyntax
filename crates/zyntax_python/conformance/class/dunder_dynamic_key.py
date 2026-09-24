# Item and operator dunders reached through a dynamic receiver with
# keys of every kind: the typed calls in view give the methods tuple
# and int keys, and the boxed calls must still take whatever they get.
class Grid(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.cells = [0] * (w * h)
        self.names = {"origin": 0}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.cells[self.names[key]]
        if isinstance(key, int):
            return self.cells[key]
        x, y = key
        return self.cells[y * self.width + x]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.cells[self.names[key]] = value
            return
        if isinstance(key, int):
            self.cells[key] = value
            return
        x, y = key
        self.cells[y * self.width + x] = value

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.names
        if isinstance(key, int):
            return 0 <= key < len(self.cells)
        x, y = key
        return 0 <= x < self.width and 0 <= y < self.height

    def __add__(self, other):
        if isinstance(other, int):
            g = Grid(self.width, self.height)
            g.cells = [c + other for c in self.cells]
            return g
        if isinstance(other, str):
            g = Grid(self.width, self.height)
            g.cells = list(self.cells)
            g.cells[self.names[other]] += 100
            return g
        x, y = other
        g = Grid(self.width, self.height)
        g.cells = list(self.cells)
        g.cells[y * self.width + x] += 1
        return g


def show(g):
    print(g.cells)


def attempt(label, thunk):
    try:
        print(label, thunk())
    except TypeError as e:
        print(label, "TypeError", e)
    except ValueError as e:
        print(label, "ValueError", e)
    except KeyError as e:
        print(label, "KeyError", e)


def main():
    g = Grid(3, 2)
    # Typed sites: tuple and int keys, an int operand.
    g[1, 1] = 5
    g[0] = 7
    print(g[1, 1], g[0], g[2, 0], (1, 1) in g, (5, 5) in g, 2 in g, 99 in g)
    show(g + 1)
    show(g + (2, 1))
    # The same through a dynamic receiver, with keys of every kind.
    boxed = [g, 0]
    d = boxed[0]
    attempt("get tuple", lambda: d[(2, 1)])
    attempt("get longer", lambda: d[(1, 1, 1)])
    attempt("get shorter", lambda: d[(1,)])
    attempt("get str", lambda: d["origin"])
    attempt("get missing str", lambda: d["nowhere"])
    attempt("get int", lambda: d[1])

    def set_tuple():
        d[(2, 1)] = 8
        return g.cells

    def set_longer():
        d[(1, 1, 1)] = 8
        return g.cells

    def set_shorter():
        d[(1,)] = 8
        return g.cells

    def set_str():
        d["origin"] = 9
        return g.cells

    def set_int():
        d[2] = 4
        return g.cells

    attempt("set tuple", set_tuple)
    attempt("set longer", set_longer)
    attempt("set shorter", set_shorter)
    attempt("set str", set_str)
    attempt("set int", set_int)
    attempt("in tuple", lambda: (0, 0) in d)
    attempt("in longer", lambda: (0, 0, 0) in d)
    attempt("in shorter", lambda: (0,) in d)
    attempt("in str", lambda: "origin" in d)
    attempt("in int", lambda: 3 in d)
    attempt("add tuple", lambda: (d + (0, 0)).cells)
    attempt("add longer", lambda: (d + (0, 0, 0)).cells)
    attempt("add shorter", lambda: (d + (0,)).cells)
    attempt("add str", lambda: (d + "origin").cells)
    attempt("add int", lambda: (d + 10).cells)


main()
