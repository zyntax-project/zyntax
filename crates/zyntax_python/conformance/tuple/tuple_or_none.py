# A value that is a tuple here and None there: kept, passed, stored,
# compared and printed, and None raising where a tuple is read.
def find(xs, key):
    best = None
    for x in xs:
        if x[0] == key:
            if best is None or x[1] < best[1]:
                best = x
    return best


def pair(i):
    if i % 3 == 0:
        return None
    return (i, i * 0.5)


def describe(r):
    if r is None:
        return "none"
    (k, v, w) = r
    return "%s %s %s" % (k, v + 1.0, w)


class Holder(object):
    def __init__(self):
        self.last = None

    def keep(self, r):
        self.last = r


xs = [(1, 2.5, "a"), (2, 0.5, "b"), (1, 1.5, "c")]
print(describe(find(xs, 1)), describe(find(xs, 2)), describe(find(xs, 3)))
r = find(xs, 1)
print(r, r is None, r is not None, bool(r), r[1], find(xs, 9), str(r), repr(find(xs, 9)))
print(find(xs, 1) == (1, 1.5, "c"), find(xs, 9) == None, [find(xs, k) for k in range(4)])
h = Holder()
for k in (1, 9, 2):
    h.keep(find(xs, k))
    print(h.last)
ps = [pair(i) for i in range(5)]
print(ps, [p for p in ps if p], {i: pair(i) for i in range(3)})
total = 0.0
for i in range(6):
    p = pair(i)
    if p is not None:
        a, b = p
        total += a + b
print(total, (None, 1), (1, None) == (1, None), [(None, i) for i in range(2)])
for bad in ("index", "unpack", "iterate", "len"):
    try:
        t = find(xs, 9)
        if bad == "index":
            print(t[0])
        elif bad == "unpack":
            a, b, c = t
        elif bad == "iterate":
            for e in t:
                pass
        else:
            print(len(t))
    except TypeError as e:
        print("TypeError", e)
