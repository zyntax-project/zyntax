# Arithmetic on a local that holds an int or a float by turns: each
# operation's result and kind on either path.
def show(v):
    print(repr(v), "float" if isinstance(v, float) else "int")


def run(flag, n, d):
    x = n
    if flag:
        x = n + 0.5
    y = d
    if not flag:
        y = d * 1.0
    show(x + y)
    show(x - y)
    show(x * y)
    show(x / y)
    show(x // y)
    show(x % y)
    show(x ** 2)
    show(x // 3)
    show(x % 3)
    show(x / 4)
    show(-7 // y)
    show(-7 % y)
    show(x // 2.0)
    show(x % 2.5)
    try:
        show(x // (y - y))
    except ZeroDivisionError as e:
        print("ZeroDivisionError:", e)
    try:
        show(x % (y - y))
    except ZeroDivisionError as e:
        print("ZeroDivisionError:", e)
    try:
        show(x / (y - y))
    except ZeroDivisionError as e:
        print("ZeroDivisionError:", e)


def bools(flag):
    b = True
    if flag:
        b = 7
    show(b + 1)
    show(b // 2)
    show(b % 2)
    show(b / 2)
    show(b ** 2)
    try:
        show(3 // (b - b))
    except ZeroDivisionError as e:
        print("ZeroDivisionError:", e)


def signs(flag):
    z = -0.0
    if flag:
        z = 0
    show(z // float("nan"))
    show(z // -7)


run(True, 7, 2)
run(False, 7, 2)
run(True, -9, 4)
run(False, -9, 4)
bools(True)
bools(False)
signs(True)
signs(False)
