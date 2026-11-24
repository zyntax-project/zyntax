class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return x + self.n


class Plain:
    pass


def f(x):
    return x


def check(v):
    return callable(v)


g = lambda a: a
add = Adder(3)
p = Plain()
print(callable(f), callable(len), callable(Adder), callable(Plain), callable(g))
print(callable(add), callable(p), callable(1), callable("s"), callable([1]), callable(None))
print(callable(add.__call__))
items = [f, add, p, 3, "x", g]
print([callable(i) for i in items])
print([check(i) for i in items])
print(callable(print), callable(int), callable(str))
m = add
print(callable(m))
