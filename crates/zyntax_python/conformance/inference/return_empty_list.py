# `return []` beside returns of a typed list is a list of that kind; on
# its own, or beside anything else, it is still a fresh empty list.
def moves(cell, vals):
    if cell < 0:
        return []
    return [(cell, v) for v in vals]


def only_empty():
    return []


def mixed(flag):
    if flag:
        return []
    return "text"


def main():
    got = moves(2, [1, 3])
    none = moves(-1, [1])
    none.append((9, 9))
    print(got, none, moves(-1, [])[:], len(moves(-1, [5])))
    first, v = moves(4, [7])[0]
    print(first + v)
    e = only_empty()
    e.append("x")
    print(e, only_empty(), mixed(True), mixed(False))
    a = moves(-1, [])
    b = moves(-1, [])
    a.append((1, 1))
    print(a, b, a is b)


main()
