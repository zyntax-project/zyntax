# min and max of several ints or floats keep the first least or greatest,
# read each argument once and in order; int() and float() of a value that
# is an int or a float take it as it is.
order = []


def note(v):
    order.append(v)
    return v


print(min(3, 1, 2), max(3, 1, 2), min(2, 2), max(-1, -1), min(5, note(4), note(6)), order)
print(min(1.5, 0.5), max(1.5, 0.5), min(0.0, -0.0), max(-0.0, 0.0), min(2.0, 2.0))
nan = float("nan")
print(min(nan, 1.0), min(1.0, nan), max(nan, 1.0), max(1.0, nan))
print(min(1, 2.5), max(1, 2.5), min(True, 0), max(3, 2, 1.5))


def clamp(v):
    return max(0, min(255, v))


print([clamp(v) for v in (-5, 0, 100, 255, 300)])


def mixed(i):
    x = 1
    if i % 3 == 1:
        x = 2.75
    elif i % 3 == 2:
        x = True
    return int(x * 10), float(x), int(x)


print([mixed(i) for i in range(4)])


def scale(c):
    r = 0
    if c > 0:
        r = c * 0.5
    return max(0, min(255, int(r * 255)))


print([scale(c) for c in (0, 1, 2, 3)])
