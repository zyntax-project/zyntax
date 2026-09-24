# Membership, comparison and call dunders reached from typed sites with
# arguments of several types, and from a dynamic receiver with the same
# arguments and with ones the methods reject.
class Interval(object):
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __contains__(self, x):
        if isinstance(x, Interval):
            return self.lo <= x.lo and x.hi <= self.hi
        return self.lo <= x <= self.hi

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.hi < other.lo
        return self.hi < other

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.lo == other.lo and self.hi == other.hi
        return False

    def __call__(self, t):
        return self.lo + (self.hi - self.lo) * t


def main():
    a = Interval(0, 10)
    b = Interval(2, 3)
    c = Interval(20, 30)
    print(5 in a, 11 in a, 2.5 in a, b in a, c in a, 5 not in c)
    print(a < c, c < a, a < 11, a < 10.5, a == b, a == Interval(0, 10), a != b, a == 3)
    print(a(0), a(1), a(0.25), b(2))
    boxed = [a, 1, "s"]
    d = boxed[0]
    print(5 in d, b in d, d < c, d == a, d(0.5))
    for probe in [3, 3.5, b, c]:
        print(probe in a, probe in d)
    try:
        print("s" in d)
    except TypeError:
        print("TypeError")
    try:
        print(d("x"))
    except TypeError:
        print("TypeError")


main()
