# A bool sentinel beside numbers keeps its own identity and truth.


def pick(flag):
    v = True
    if flag:
        v = 0.5
    return v


def best_of(xs):
    best = False
    for v in xs:
        if not best or v < best:
            best = v
    return best


def truth():
    for v in [0, 0.0, -0.0, float("nan"), None, False, True, 3, 2.5]:
        q = 1
        q = v
        print(bool(q), not q, q)
        if q:
            print("true")
        while q:
            q = 0
        print(q if q else "falsy")


r = pick(0)
print(r, isinstance(r, bool), r is True, r == 1)
r = pick(1)
print(r, isinstance(r, bool))
print(best_of([2.5, 1.5, 3.0]))
print(best_of([]))
truth()
