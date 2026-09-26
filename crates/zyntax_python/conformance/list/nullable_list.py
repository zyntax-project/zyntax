# A local or a field that is a list or None: None-preserving uses keep
# None; reads through None raise CPython's TypeError.
class Done:
    def __init__(self, count, empty=False):
        self.count = count
        self.cells = None if empty else [[0, 1] for i in range(count)]

    def clone(self):
        ret = Done(self.count, True)
        ret.cells = [self.cells[i][:] for i in range(self.count)]
        return ret

    def set_done(self, i, v):
        self.cells[i] = [v]

    def already_done(self, i):
        return len(self.cells[i]) == 1


def f(x=None):
    if not x:
        return "empty"
    return x + 1


def g(x=None):
    return x or 5


def reads(d):
    try:
        print(d.cells[0])
    except TypeError as e:
        print("subscript TypeError", e)
    try:
        d.cells[0] = 1
    except TypeError as e:
        print("store TypeError", e)


def locals_of_both(n):
    xs = None
    if n > 1:
        xs = [n, n + 1]
    if xs:
        print("truthy", xs[0], len(xs))
    if xs is None:
        print("is None")
    if xs is not None:
        xs[0] = 9
        print("not None", xs)
    ys = [] if n > 5 else None
    print(xs, ys, [xs, ys], str(xs), repr(ys), f"{xs}|{ys}")
    try:
        print(ys[0])
    except (TypeError, IndexError) as e:
        print("error", e)
    try:
        ys[0] = 1
    except (TypeError, IndexError) as e:
        print("error", e)
    return xs


def main():
    d = Done(2, True)
    c = d.cells
    print(c, c is None, c == None, bool(c), [c])
    print(f(), f(0), f(3), g(), g(0), g(7))
    reads(d)
    try:
        d.set_done(0, 3)
    except TypeError as e:
        print("TypeError", e)
    full = Done(3)
    full.set_done(1, 4)
    copy = full.clone()
    copy.cells[0].append(5)
    print(full.cells, copy.cells, full.already_done(1), copy.already_done(0))
    held = full.cells
    full.cells = None
    print(held, full.cells, held is not None)
    print(locals_of_both(1), locals_of_both(2), locals_of_both(7))


main()
