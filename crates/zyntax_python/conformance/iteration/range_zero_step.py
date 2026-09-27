xrange = range


def literal_zero():
    try:
        for q in range(1, 2, 0):
            print("ran", q)
    except ValueError as e:
        print("ValueError", e)


def runtime_step(s):
    n = 0
    try:
        for q in range(1, 10, s if s >= 0 else -s):
            n += q
        print("sum", n, q)
    except ValueError as e:
        print("ValueError", e)


def aliased(s):
    try:
        for q in xrange(1, 2, s):
            print("ran", q)
    except ValueError as e:
        print("ValueError", e)


def negative(s):
    out = []
    for q in range(10, 0, s):
        out.append(q)
    print(out)


def comprehension(s):
    try:
        print([q * 2 for q in range(0, 6, s)])
    except ValueError as e:
        print("ValueError", e)


def generator(s):
    try:
        print(sum(q for q in range(0, 6, s)))
    except ValueError as e:
        print("ValueError", e)


def order():
    seen = []

    def arg(v):
        seen.append(v)
        return v

    try:
        for q in range(arg(1), arg(5), arg(0)):
            pass
    except ValueError:
        print("order", seen)


def uncaught(s):
    for q in range(0, 3, s):
        print(q)


literal_zero()
runtime_step(0)
runtime_step(3)
runtime_step(-2)
aliased(0)
aliased(1)
negative(-3)
negative(-1)
comprehension(0)
comprehension(2)
generator(0)
generator(-1)
generator(2)
order()
try:
    uncaught(0)
except ValueError as e:
    print("outer", e)
uncaught(1)
