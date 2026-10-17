# A value that holds one of several module functions, functions passed
# only to parameters that call them, and ones that reach code that may
# call them with anything.
def double(x):
    return x * 2


def half(x):
    return x / 2


def apply(f, xs):
    return [f(v) for v in xs]


def pick(flag):
    return double if flag else half


print(apply(double, [1, 2]), apply(half, [1, 2]))
g = pick(True)
print(g(5), pick(False)(5))
for f in (double, half):
    print(f(3))


def row(i_u):
    i, u = i_u
    total = 0
    for j, u_j in enumerate(u):
        total += (i + j) * u_j
    return total


def column(i_u):
    i, u = i_u
    total = 0
    for j, u_j in enumerate(u):
        total += u_j / (i + j + 1)
    return total


def times(func, u):
    return [func((i, u)) for i in range(len(u))]


u = [1] * 4
for _ in range(2):
    u = times(column, times(row, u))
print(u)


def shout(s):
    return s.upper()


def keep(f, xs):
    held = [f]
    return [held[0](x) for x in xs]


def twice(x):
    return x + x


print(keep(twice, [1, 2.5]), keep(twice, ["a"]), keep(shout, ["b"]))
