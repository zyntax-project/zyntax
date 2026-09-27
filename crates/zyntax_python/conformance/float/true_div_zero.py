# ZeroDivisionError on each numeric path, typed and dynamic, with
# CPython's text for each operation.
def typed(a, b, x, y):
    # a, b ints; x, y floats
    for name, f in [
        ("int / int", lambda: a / b),
        ("float / int", lambda: x / b),
        ("int / float", lambda: a / y),
        ("float / float", lambda: x / y),
        ("int // int", lambda: a // b),
        ("int % int", lambda: a % b),
        ("float // float", lambda: x // y),
        ("float % float", lambda: x % y),
        ("int // float", lambda: a // y),
        ("float % int", lambda: x % b),
        ("divmod int", lambda: divmod(a, b)),
        ("divmod float", lambda: divmod(x, y)),
        ("0.0 ** -1", lambda: y ** -1),
        ("0.0 ** -0.5", lambda: y ** -0.5),
    ]:
        try:
            print(name, f())
        except ZeroDivisionError as e:
            print(name, "ZeroDivisionError:", e)


def direct(a: int, b: int, x: float, y: float):
    try:
        print(a / b)
    except ZeroDivisionError as e:
        print("direct int /", e)
    try:
        print(x / y)
    except ZeroDivisionError as e:
        print("direct float /", e)
    try:
        print(a % b)
    except ZeroDivisionError as e:
        print("direct int %", e)
    try:
        print(x // y)
    except ZeroDivisionError as e:
        print("direct float //", e)
    try:
        print(x % y)
    except ZeroDivisionError as e:
        print("direct float %", e)
    try:
        print(divmod(x, y))
    except ZeroDivisionError as e:
        print("direct divmod", e)
    try:
        print(y ** -2)
    except ZeroDivisionError as e:
        print("direct pow", e)
    # Nonzero divisors compute as before.
    print(a / 2, x / 2.0, x // 2.0, x % 2.0, divmod(x, 2.0), 2.0 ** -1)


typed(7, 0, 7.5, 0.0)
direct(7, 0, 7.5, 0.0)
direct(7, 0, -7.5, -0.0)
print(7 / 2, 7.5 // 2, -7.5 % 2, divmod(7.5, 2))

vals = [1, 0, 2.5, 0.0, True, False]
for u in vals:
    for v in vals:
        for op in ("/", "//", "%"):
            try:
                if op == "/":
                    r = u / v
                elif op == "//":
                    r = u // v
                else:
                    r = u % v
                print(u, op, v, r)
            except ZeroDivisionError as e:
                print(u, op, v, "ZeroDivisionError:", e)
