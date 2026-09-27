# A None default replaced under an `is None` test, and rebindings
# a later store or a loop keeps reaching.

def a(x=None):
    if x is None:
        x = 10
    return x + 1

def b(x=None):
    if x == None:
        x = [1, 2]
    return len(x)

def c(x=None):
    if x is None:
        return -1
    return x * 2

def f(x=None):
    if x is None:
        x = 1
    for i in range(2):
        print(x)
        x = None

def g(x=None):
    for i in range(0):
        if x is None:
            x = 1
    print(x)

def h(x=None):
    ys = list(v + (0 if x is None else x) for v in range(3))
    if x is None:
        x = 5
    return ys, x

class P:
    def tp(self, p, trafo=None):
        if trafo is None:
            trafo = self.rt()
        return p + trafo[0] * trafo[1]
    def rt(self):
        return 3, 4

print(a(), a(5), b(), b("abc"), c(), c(4))
f()
g()
print(h(), h(2))
p = P()
print(sum(p.tp(i) for i in range(5)))
