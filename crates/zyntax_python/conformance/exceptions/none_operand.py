# None in arithmetic and ordering, through a local that may hold it:
# TypeError with CPython's text; equality never raises.
def f(flag):
    x = None
    if flag:
        x = 3
    for name, g in [
        ("+", lambda: x + 1),
        ("-", lambda: 1 - x),
        ("*", lambda: x * 2),
        ("/", lambda: x / 2),
        ("//", lambda: x // 2),
        ("%", lambda: x % 2),
        ("**", lambda: x ** 2),
        ("&", lambda: x & 1),
    ]:
        try:
            print(name, g())
        except TypeError as e:
            print(name, "TypeError:", e)
    try:
        print(x + 1)
    except TypeError as e:
        print("TypeError:", e)
    try:
        print(x < 1)
    except TypeError as e:
        print("TypeError:", e)
    print(x == None, x != 3, x == 3)


f(True)
f(False)
