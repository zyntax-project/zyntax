# Parameters and results that hold an int here and a float or None there.
def acc(xs):
    total = 0
    for x in xs:
        total += x
    return total
def half(v):
    return v / 2
def pick(flag):
    if flag:
        return 1
    return 2.5
def maybe(n):
    if n > 3:
        return n
def scale(v, k):
    return v * k
print(acc([]), acc([1.5, 2.0]), repr(acc([])), repr(acc([0.5])))
print(half(acc([1.0, 3.0])), half(3))
print(pick(True), pick(False), repr(pick(True)))
print(maybe(5), maybe(1), maybe(2) is None)
for v in (1, 2.5, True):
    print(scale(v, 3), repr(scale(v, 2)))
print(scale(pick(True), pick(False)))
