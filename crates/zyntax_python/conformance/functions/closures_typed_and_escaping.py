# A lambda or nested def called where its value is known, and the same
# value once it has escaped into a list, a sort key, or another
# function's parameter.

def counter():
    n = 0
    def inc():
        nonlocal n
        n += 1
        return n
    return inc

def make_adder(k):
    def add(x):
        return x + k
    return add

def fact_maker():
    def fact(n):
        if n <= 1:
            return 1
        return n * fact(n - 1)
    return fact

def twice(f, v):
    return f(f(v))

def use(f):
    return f(10)

def main():
    f = lambda x: x + 1
    print(f(1))
    fs = [f]
    print(fs[0](2))
    print(fs[0](2.5))
    g = lambda a, b: a * b
    print(g(2, 3))
    print(g("ab", 2))

    c = counter()
    c(); c()
    print(c())
    add5 = make_adder(5)
    print(add5(10))
    addf = make_adder(0.5)
    print(addf(1))
    print(fact_maker()(10))

    key = lambda p: p[1]
    xs = [(1, 3), (2, 1), (3, 2)]
    print(sorted(xs, key=key))
    h = lambda: 42
    print(h())
    sq = lambda x: x * x
    print(sq(3), sq(1.5))

    def maybe(x):
        if x > 0:
            return x
    print(maybe(3))
    print(maybe(-3))
    def noop(x):
        pass
    print(noop(1))
    def apply(fn, v):
        return fn(v)
    print(apply(lambda z: z * 2, 21))
    print(apply(lambda z: z + "!", "hi"))

    double = lambda x: x * 2
    print(twice(double, 3))
    print(use(lambda v: v + 1))
    print(use(lambda v: v * 1.5))
    total = 0
    for i in [1, 2, 3]:
        total = double(total) + i
    print(total)

main()
