import time


def step(t: tuple[int, int, int], k: int) -> tuple[int, int, int]:
    a, b, c = t
    # Recursive, so no caller inlines it: the calls reach its own entry.
    if k % 5 == 0 and k > 0:
        return step((b, c, a), k - 1)
    return (b + k, c, (a * 7 + k) % 1000003)


def run(n: int) -> int:
    total = 0
    for i in range(n):
        t = step((i, i * 3, total % 1000), i)
        total = (total + t[0] + t[2]) % 1000000007
    return total


def main() -> None:
    first = run(20000)
    # Let the background LLVM promotion of step finish.
    until = time.time() + 1.0
    while time.time() < until:
        pass
    second = run(20000)
    print(first, second)


main()
