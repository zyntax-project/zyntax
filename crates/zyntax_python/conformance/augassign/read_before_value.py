# test_augassign: the target's item, attribute or global is read before
# the right-hand side runs; a list grows by values of any kind.
log = []
xs = [1, 2]


def bump():
    log.append("bump")
    xs[0] = 100
    if len(log) > 10:
        raise ValueError("too many")
    return 1


xs[0] += bump()
print(xs, log)


class Box:
    def __init__(self):
        self.n = 5


b = Box()


def rebox():
    log.append("rebox")
    b.n = 50
    if len(log) > 10:
        raise ValueError("too many")
    return 2


b.n *= rebox()
print(b.n, log)

total = 3


def retotal():
    global total
    log.append("retotal")
    total = 30
    if len(log) > 10:
        raise ValueError("too many")
    return 4


total += retotal()
print(total, log)

ys = [1, 2]
same = ys
ys += (3,)
print(ys, same, ys is same)
ys += ("a", 2.5)
print(ys)
zs = ["p"]
zs += (1, "q")
print(zs)
try:
    ys += 5
except TypeError:
    print("TypeError")


def scale(v, k):
    alias = v
    v *= k
    return alias is v, v


def grow(v, w):
    alias = v
    v += w
    return alias is v, v


print(scale([1, 2], 2), scale([1.5], 0), scale(3, 2), scale("ab", 2), scale(2.5, 2))
print(grow([1], [2]), grow(["a"], ("b",)), grow(1, 2), grow(1.5, 2.0), grow("a", "b"))
