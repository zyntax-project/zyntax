# Unary minus and plus of a local that holds an int, a float or a bool
# by turns.
def show(v):
    print(repr(v), "float" if isinstance(v, float) else "int")


def run(flag, n):
    x = n
    if flag:
        x = n * 0.5
    show(-x)
    show(+x)
    show(-x + 1)
    show(-(-x))


def bools(flag):
    b = True
    if flag:
        b = -3
    show(-b)
    show(+b)


def floats(flag):
    b = False
    if flag:
        b = 2.5
    show(-b)
    show(+b)


def none(flag):
    x = None
    if flag:
        x = 4
    try:
        print(-x)
    except TypeError as e:
        print("TypeError")


run(True, 7)
run(False, 7)
run(False, -(2**62))
bools(True)
bools(False)
floats(True)
floats(False)
none(True)
none(False)


def boxed(v):
    try:
        print(-v)
    except TypeError as e:
        print("TypeError:", e)


for v in [0.0, -0.0, 3, 2.5, True, None, "s"]:
    boxed(v)
