# A generator raising after a yield, consumed by a function with no
# other fallible call, caught in a third.
def gen(n: int):
    for i in range(n):
        if i == 2:
            raise ValueError("boom")
        yield i


def consume(g):
    total = 0
    for v in g:
        total += 1
    return total


def run():
    try:
        g = gen(5)
        n = consume(g)
        return n
    except ValueError as e:
        return -1


for _ in range(3):
    print(run())


def tidy(n: int):
    try:
        yield n
        raise KeyError("after yield")
    finally:
        print("generator finally")


def drain() -> int:
    c = 0
    try:
        for v in tidy(4):
            c += v
    except KeyError as e:
        print("KeyError", e)
    return c


print(drain())
