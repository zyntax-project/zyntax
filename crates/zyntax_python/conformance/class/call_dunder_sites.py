# Instances called through `__call__`: as a local, a list element and an
# attribute; a subclass overriding it; a receiver of no known class.


class Line(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, t):
        return self.a + (self.b - self.a) * t


class Doubled(Line):
    def __call__(self, t):
        return 2 * (self.a + (self.b - self.a) * t)


class Holder(object):
    def __init__(self, f):
        self.f = f

    def at(self, t):
        f = self.f
        return f(t)


line = Line(1.0, 3.0)
print(line(0), line(0.5), line(1))
lines = [Line(0, 10), Line(2.0, 4.0)]
print([lines[i](0.25) for i in range(len(lines))])
h = Holder(Line(5, 7))
print(h.at(2), h.at(0.5))


def run(spl, ts):
    total = 0.0
    for t in ts:
        total += spl(t)
    return total


print(run(Line(0.0, 1.0), [0.0, 0.5, 1.0]))
print(run(Doubled(0.0, 1.0), [0.0, 0.5, 1.0]))
mixed = [Line(0, 1), Doubled(0, 1)]
print([m(0.5) for m in mixed])


def call_any(f, x):
    return f(x)


print(call_any(Line(1, 2), 3), call_any(lambda v: v * 10, 3), call_any(abs, -3))
