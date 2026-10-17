# Operator, comparison and call methods of a class whose instances are
# never held as dynamic values are typed by their calls; a class whose
# instances are held dynamically still answers every operand.
class V(object):
    def __init__(self, x=0.0):
        self.x = x

    def __add__(self, other):
        return V(self.x + other.x)

    def __sub__(self, other):
        return self + other * -1

    def __mul__(self, other):
        return V(self.x * other)
    __rmul__ = __mul__

    def __lt__(self, other):
        return self.x < other.x

    def __eq__(self, other):
        return isinstance(other, V) and self.x == other.x

    def __call__(self, k):
        return self.x * k


a = V(1.5)
b = V(2.0)
c = a - b
print(c.x, (a + b).x, (2 * a).x, (a * 3).x, a < b, b < a, a == V(1.5), a == b, a(4))
print((a - V(0.5)).x, (b * 2 - a).x, (-1 * b).x)


class Ordered(object):
    def __init__(self, k):
        self.k = k

    def __lt__(self, other):
        return self.k < other.k


print([o.k for o in sorted([Ordered(3), Ordered(1), Ordered(2)])], max([Ordered(4), Ordered(9)]).k)


class Held(object):
    def __init__(self, n):
        self.n = n

    def __add__(self, other):
        if isinstance(other, Held):
            return Held(self.n + other.n)
        return Held(self.n + other)

    def __radd__(self, other):
        return Held(other + self.n)

    def __lt__(self, other):
        return self.n < (other.n if isinstance(other, Held) else other)

    def __repr__(self):
        return "Held(%r)" % (self.n,)


mixed = [Held(1), 2, Held(3.5), "s"]
for v in mixed[:3]:
    print(v + 1, 1 + v, v + Held(10), v < 2)
print(Held(1) + 2.5, sum([Held(1), Held(2)], Held(0)), sorted([Held(3), Held(1)]))
