# Fields that hold an int here and a float or None there.
class V(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def add(self, o):
        return V(self.x + o.x, self.y + o.y)
    def scale(self, k):
        return V(self.x * k, self.y * k)
    def __repr__(self):
        return "V(%r, %r)" % (self.x, self.y)
class Holder(object):
    def __init__(self):
        self.best = None
    def offer(self, v):
        if self.best is None or v > self.best:
            self.best = v
a = V(1, 2)
b = V(0.5, 1.5)
c = a.add(b).scale(2)
print(a, b, c, repr(c.x), c.x + c.y)
d = V()
d.x += 3
d.y = d.y + 0.25
print(d, d.x // 2, d.y * 4)
h = Holder()
print(h.best)
for v in (3, 1.5, 7, 2):
    h.offer(v)
print(h.best, h.best + 1)
o = [a, b, c][1]
print(o.x, o.x == 0.5)
o.x = 9
print(o, o.x + 1)


# One instance's number field copied into another's, both reached
# through method calls.
class Box(object):
    def __init__(self, v):
        self.value = v


class Pair(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def inp(self):
        return self.a

    def out(self):
        return self.b

    def run(self):
        self.out().value = self.inp().value


first, second = Box(1), Box(2.5)
Pair(first, second).run()
print(first.value, second.value)
