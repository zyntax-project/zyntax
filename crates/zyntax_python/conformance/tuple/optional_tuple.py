# A tuple or None: the None-sentinel search, its tests, and what each
# operation on None raises.

def first(items):
    result = None
    for i in items:
        x = i[1]
        if x > 0:
            if result is None or x < result[1]:
                result = i
    return result


def pair(k):
    if k:
        return (k, k + 1)
    return None


def main():
    r = first([(1, 5.0), (2, 3.0), (3, -1.0), (4, 4.0)])
    print(r)
    if r is not None:
        a, b = r
        print(a, b)
    n = first([(1, -1.0)])
    print(n, n is None, n == None, r == None, r is not None)
    print(r == (2, 3.0), (2, 3.0) == r, r != (2, 3.0), r == (2, 3.5), n == (2, 3.0))
    print(bool(n), bool(r), not n, not r)
    if n:
        print("unreachable")
    for op in range(5):
        try:
            if op == 0:
                n[0]
            elif op == 1:
                a, b = n
            elif op == 2:
                len(n)
            elif op == 3:
                for q in n:
                    print(q)
            else:
                n.count(1)
        except TypeError as e:
            print("TypeError", e)
        except AttributeError as e:
            print("AttributeError", e)
    x, y = pair(3)
    print(x + y)
    try:
        x, y = pair(0)
    except TypeError as e:
        print("TypeError", e)
    total = 0
    for k in range(4):
        p = pair(k)
        if p is None:
            continue
        u, v = p
        total += u * v
    print(total)


main()
