# A return, break or continue inside finally discards the exception
# that was propagating.
def f() -> int:
    try:
        d = {"a": 1}
        return d["missing"]
    finally:
        return 7


print(f())


def g() -> int:
    n = 0
    for i in range(3):
        try:
            if i == 1:
                raise ValueError("one")
            n += 1
        finally:
            continue
    return n


print(g())


def h() -> int:
    n = 0
    while True:
        try:
            n += 1
            raise KeyError("stop")
        finally:
            break
    return n


print(h())


def keep() -> int:
    try:
        raise ValueError("kept")
    finally:
        print("finally without leaving")


try:
    keep()
except ValueError as e:
    print("ValueError:", e)
print("done")
