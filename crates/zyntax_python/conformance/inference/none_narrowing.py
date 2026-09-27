# Numbers or None read where a test has shown they are not None: in
# the branch, the right of and/or, after an early exit, and where a
# later store or a try ends what the test showed.

def best(xs):
    b = None
    for x in xs:
        if b is None or x < b:
            b = x
    return b

def main_minmax():
    lo = None
    hi = None
    s = 0.0
    for i in range(30):
        v = (i * 7 % 11) * 0.5
        if lo is None or v < lo:
            lo = v
        if hi is not None and v <= hi:
            s += hi - v
        else:
            hi = v
    print(lo, hi, s)
    print(best([3, 1, 2]), best([]))
    m = None
    for i in range(3):
        if m is not None:
            m = m + i
        else:
            m = i
    print(m)

main_minmax()

def only_none(k):
    x = None
    if x is not None:
        y = x
        print(y + 1)
    for i in range(k):
        if x is None:
            continue
        print(x * 2)
    print(x)

def later_store(k):
    x = None
    for i in range(k):
        if x is not None:
            x = x + 1
            print(x)
            x = None
            print(x)
        else:
            x = i
    print(x)

def comp(k):
    x = None
    if k:
        x = 2.5
    if x is not None:
        ys = [x * j for j in range(3)]
        zs = [x for x in range(2)]
        print(ys, zs, x)
    return x

def early(n):
    best = None
    for v in [3.5, 1.5, 2.5][:n]:
        if best is not None and v >= best:
            continue
        best = v
    if best is None:
        return -1.0
    return best * 2

def trying(k):
    x = None
    if k:
        x = 3
    if x is not None:
        try:
            print(x + 1)
        except ValueError:
            pass
        print(x - 1)
    return x

def bools(k):
    b = None
    if k > 1:
        b = k > 2
    if b is not None:
        print(b, not b, b + 1)
    return b

only_none(3)
later_store(5)
print(comp(0), comp(1))
print(early(0), early(3), early(1))
print(trying(0), trying(1))
print(bools(0), bools(2), bools(3))


def list_of_later_none():
    v = 1.5
    out = []
    for i in range(3):
        out.append(v)
        if i == 1:
            v = None
    return out


print(list_of_later_none())
