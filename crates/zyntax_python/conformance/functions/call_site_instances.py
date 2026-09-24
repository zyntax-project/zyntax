# One function called with arguments of several types: each call gets
# the result its own arguments give, whether or not the function has an
# instance for them, and a body that raises does so for each.
def add(a, b):
    return a + b


def scale(xs, k):
    return [x * k for x in xs]


def move_to_front(l, c):
    l[:] = l[c:c + 1] + l[0:c] + l[c + 1:]


def first_or(xs, default):
    if len(xs) == 0:
        return default
    return xs[0]


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("nothing to divide by")
    return a / b


def describe(x):
    return "<" + str(x) + ">"


class Pair(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def swap(self):
        return Pair(self.b, self.a)

    def combine(self, other, k):
        return Pair(self.a + other.a * k, self.b + other.b * k)


def main():
    print(add(1, 2), add(1.5, 2), add("x", "y"), add([1], [2]), add(True, 1))
    print(scale([1, 2], 3), scale([1.5], 2), scale(["a"], 2))
    xs = [1, 2, 3]
    move_to_front(xs, 2)
    ys = ["a", "b", "c"]
    move_to_front(ys, 1)
    zs = [1.5, 2.5]
    move_to_front(zs, 1)
    print(xs, ys, zs)
    print(first_or([], 0), first_or([1], 0), first_or([], "none"), first_or(["a"], None))
    print(divide(1, 2), divide(3.0, 2), divide(7, 2.0))
    for a, b in [(1, 0), (1.0, 0), (2, 0.0)]:
        try:
            print(divide(a, b))
        except ZeroDivisionError as e:
            print("ZeroDivisionError", e)
    print(describe(1), describe(2.5), describe("s"), describe([1]), describe(None))
    # More signatures than a function gets instances for.
    print(add(1, 2.0), add(2.0, 1), add((1,), (2,)), add(1, True))
    p = Pair(1, 2)
    q = Pair(0.5, 0.25)
    print(p.combine(p, 2).a, p.combine(q, 2).b, p.combine(q, 0.5).a, p.swap().a)


main()
