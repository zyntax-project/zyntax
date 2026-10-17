# A field bound to an empty list and passed to a function: the function
# reads it, writes into it, or keeps it.
def total(xs):
    t = 0
    for x in xs:
        t += x
    return t


def fill(xs, n):
    for i in range(n):
        xs.append(i * 0.5)


kept = []


def keep(xs):
    kept.append(xs)


class Box(object):
    def __init__(self, n):
        self.ints = []
        for i in range(n):
            self.ints.append(i)
        self.sum = total(self.ints)
        self.floats = []
        fill(self.floats, n)
        self.held = []
        keep(self.held)
        self.held.append("s")


b = Box(3)
print(b.ints, b.sum, b.floats, b.held, kept)
kept[0].append(1)
print(b.held)
