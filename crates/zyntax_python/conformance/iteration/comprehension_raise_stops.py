# An element that raises ends the comprehension before anything uses it.
def r(a: int) -> int:
    if a == 2:
        raise ValueError("two")
    return a * 10


def h(xs: list[int]) -> int:
    print("h entered")
    return len(xs)


def leak2(xs: list[int]) -> int:
    return h([r(a) for a in xs])


try:
    print(leak2([1, 3]))
    print(leak2([1, 2, 3]))
except ValueError as e:
    print("ValueError:", e)


def comp_int() -> int:
    p, q = [int(a) for a in ["1", "2", "3", "x"]]
    return p + q


try:
    comp_int()
except ValueError as e:
    print("ValueError:", e)

try:
    print(h([r(a) for a in [2]]))
except ValueError as e:
    print("module ValueError:", e)

try:
    d = {a: r(a) for a in [1, 2]}
    print("dict", d)
except ValueError as e:
    print("dict ValueError:", e)
