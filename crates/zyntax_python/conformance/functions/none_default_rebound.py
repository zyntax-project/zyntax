# A parameter defaulting to None and set at the top of the body when it
# is None: callers that pass it and callers that leave it out.
def pick(xs, first=None):
    if first is None:
        first = xs[0]
    return first * 2


def split(s, sep=None, keep=None):
    if sep is None:
        sep = " "
    if keep is None:
        keep = []
    keep.extend(s.split(sep))
    return keep


def scale(v, k=None):
    if k is None:
        k = 1 - v
    total = 0
    for i in range(3):
        total += v * k + i
    k = k + 1
    return total, k


def seen(x, acc=None):
    print("before", acc)
    if acc is None:
        acc = [x]
    return acc


def closure(n, step=None):
    if step is None:
        step = 2
    f = lambda i: i * step
    return [f(i) for i in range(n)]


print(pick([3, 4]), pick([3, 4], 5), pick([1.5]))
print(split("a b c"), split("a,b", ","), split("x y", None, ["w"]))
print(scale(0.25), scale(0.5, 2.0), scale(2))
print(seen(1), seen(2, [0]))
print(closure(3), closure(2, 5))
