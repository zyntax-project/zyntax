# A loop that drains a fresh generator on every pass: each step of the
# inner loop holds the generator's next item as an optional value.
def three(i: int):
    yield i
    yield i + 1
    yield i + 2


def main(n: int) -> int:
    total = 0
    for i in range(n):
        for v in three(i):
            total += v
    return total


print(main(20000))
