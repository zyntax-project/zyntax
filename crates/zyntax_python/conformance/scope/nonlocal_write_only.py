# A nested function that only writes a nonlocal still shares it, and
# what it writes may be None where the outer body held an instance.
class C:
    def __init__(self, v):
        self.v = v


def counter():
    n = 0

    def bump():
        nonlocal n
        n = n + 1

    bump()
    bump()
    return n


def replace(t):
    def inner():
        nonlocal t
        t = 5

    v = t
    inner()
    return v + t


def drop(t):
    def clear():
        nonlocal t
        t = None

    v = t.v
    clear()
    try:
        return v + t.v
    except AttributeError:
        return -3


def is_none_after(t):
    def clear():
        nonlocal t
        t = None

    clear()
    return t is None


def swap(t):
    def other():
        nonlocal t
        t = C(9)

    other()
    return t.v


print(counter())
print(replace(4))
print(drop(C(4)))
print(is_none_after(C(1)))
print(swap(C(4)))
