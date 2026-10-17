# min() and max() of ints and floats, and of values that are either.
def f(xs):
    acc = 0
    for x in xs:
        acc = acc + x
    a = min(1, acc)
    b = max(acc, 2, 0.5)
    c = min(acc, None) if False else min(acc, 3)
    return a, b, c, min(2, 1.0), max(1, 1.0), min(1.0, 1), max(True, 0)
print(f([0.25, 0.5]), f([]), f([3, 4]))
