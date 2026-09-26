# A finally clause runs its calls while an exception propagates, and the
# exception carries on once the clause ends.
def cleanup(n: int) -> int:
    print("cleanup start", n)
    m = n + 1
    print("cleanup end", m)
    return m


def work(xs: list[int]) -> int:
    try:
        return xs[5]
    finally:
        cleanup(2)
        cleanup(len(xs))
        print("finally tail")


try:
    work([1, 2])
    print("not reached")
except IndexError:
    print("IndexError caught")


def nested(k: int) -> int:
    try:
        try:
            raise ValueError("inner")
        finally:
            print("inner finally", cleanup(k))
    finally:
        print("outer finally", cleanup(k + 10))
    return 0


try:
    nested(1)
except ValueError as e:
    print("ValueError:", e)

# A handler that raises again still runs the finally before leaving.
def rethrow() -> int:
    try:
        raise KeyError("k")
    except KeyError:
        raise ValueError("from handler")
    finally:
        print("finally after handler", cleanup(0))


try:
    rethrow()
except ValueError as e:
    print("ValueError:", e)

# At module level too.
try:
    try:
        [1][3]
    finally:
        print("module finally", cleanup(7))
except IndexError:
    print("module IndexError")
