# A number that is an int or a float formats as the kind it holds.


def value(flag):
    x = 2
    if flag:
        x = 2.75
    return x


def show(flag):
    x = 2
    if flag:
        x = 2.75
    print(x, str(x), repr(x), "%s" % x, "%r" % x, f"{x}", f"{x!r}")
    print("%d" % x, "%5d|%i" % (x, x), "%.1f" % x, "%g" % x)


show(0)
show(1)
n = None
m = 1
n = m
print(n)
