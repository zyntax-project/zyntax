# A function value keeps its defaults: a module function passed as a
# value, a nested def whose default reads the enclosing scope, a lambda
# with a default, and a default naming a module variable evaluated in a
# call from another function.

SYSTEM = [1, 2, 3]


def main(n, ref="sun"):
    print(n, ref)


def invoke(f):
    f(1)
    f(2, "earth")


def outer(k):
    def f(x, y=k):
        return x + y
    return f


def energy(bodies=SYSTEM, e=0.0):
    for m in bodies:
        e += m
    return e


def twice():
    return energy() + energy([10, 20])


invoke(main)
g = outer(10)
print(g(1), g(1, 2))
h = lambda a, b=3: a * b
print(h(2), h(2, 5))
fs = [main]
fs[0](7)
try:
    g()
except TypeError:
    print("TypeError")
try:
    g(1, 2, 3)
except TypeError:
    print("TypeError")
print(twice())
