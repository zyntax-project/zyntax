# too many values to unpack gives the count for a list, tuple or dict
# source only


def dyn(v):
    try:
        a, b = v
        print(a, b)
    except ValueError as e:
        print("ValueError", e)


for v in [[1, 2, 3], (1, 2, 3), {1: 2, 3: 4, 5: 6}, {1, 2, 3}, "abc", [1], (1,), "a", {1}, {7: 8}]:
    dyn(v)


def typed():
    for f in (lambda: [1, 2, 3], lambda: (1, 2, 3)):
        try:
            a, b = f()
        except ValueError as e:
            print(e)
    try:
        a, b = "xyz"
    except ValueError as e:
        print(e)
    try:
        a, b = {1, 2, 3}
    except ValueError as e:
        print(e)
    try:
        a, b = {1: 1, 2: 2, 3: 3}
    except ValueError as e:
        print(e)
    s = "pqr"
    try:
        a, b = s
    except ValueError as e:
        print(e)
    xs = [1, 2, 3]
    try:
        a, b = xs
    except ValueError as e:
        print(e)
    try:
        x, y = "q"
    except ValueError as e:
        print(e)


typed()
