# A raise inside finally replaces the exception that was propagating.
def f() -> int:
    try:
        raise ValueError("original")
    finally:
        raise KeyError("replacement")


try:
    f()
except KeyError as e:
    print("KeyError", e)
except ValueError as e:
    print("ValueError", e)


def helper(x: int) -> int:
    if x > 0:
        raise IndexError("from helper")
    return x


def g(x: int) -> int:
    try:
        raise ValueError("first")
    finally:
        helper(x)


try:
    g(1)
except IndexError as e:
    print("IndexError", e)
except ValueError as e:
    print("ValueError", e)

try:
    g(0)
except IndexError as e:
    print("IndexError", e)
except ValueError as e:
    print("ValueError", e)


# Caught inside the finally: the original carries on.
def h() -> int:
    try:
        raise ValueError("carried")
    finally:
        try:
            helper(5)
        except IndexError as e:
            print("caught in finally:", e)


try:
    h()
except ValueError as e:
    print("ValueError", e)
