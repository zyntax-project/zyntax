# Operators on a value that is an instance of one of several classes,
# and tuples holding such values.
class Vec(object):
    def __init__(self, x):
        self.x = x

    def __add__(self, other):
        return Vec(self.x + other.x)

    def __sub__(self, other):
        if isinstance(other, Pt):
            return Pt(self.x - other.x)
        return Vec(self.x - other.x)

    def __mul__(self, k):
        return Vec(self.x * k)

    def __rmul__(self, k):
        return Vec(k * self.x)

    def __lt__(self, other):
        return self.x < other.x

    def __repr__(self):
        return "Vec(%r)" % (self.x,)


class Pt(object):
    def __init__(self, x):
        self.x = x

    def __add__(self, other):
        return Pt(self.x + other.x)

    def __sub__(self, other):
        if isinstance(other, Pt):
            return Vec(self.x - other.x)
        return Pt(self.x - other.x)

    def __lt__(self, other):
        return self.x < other.x

    def __repr__(self):
        return "Pt(%r)" % (self.x,)


def pick(i):
    if i % 2:
        return Vec(float(i))
    return Pt(float(i))


for i in range(4):
    a = pick(i)
    b = pick(i + 1)
    print(a + b, a - b, b - a, a < b)
    try:
        print(a * 2, 3 * a)
    except TypeError as e:
        print("TypeError", e)


class Box(object):
    def __init__(self):
        self.items = []

    def add(self, o, w):
        self.items.append((o, w))

    def total(self):
        t = 0.0
        for (o, w) in self.items:
            t += o.x * w
        return t


box = Box()
box.add(Vec(1.0), 2.0)
box.add(Pt(3.0), 0.5)
print(box.total(), box.items, [o for (o, w) in box.items], box.items[0] == box.items[0])
pairs = [(pick(i), i) for i in range(3)]
print(pairs, sorted(p for (p, i) in pairs))
