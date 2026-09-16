import time


def grow(xs: list[int], n: int) -> int:
    for i in range(n):
        xs.append(i)
    return len(xs)


def main() -> None:
    xs: list[int] = []
    alias = xs
    first = grow(xs, 5000)
    # Let the background LLVM promotion finish before entering grow again.
    until = time.time() + 1.0
    while time.time() < until:
        pass
    second = grow(xs, 5000)
    print(first, second, len(xs), len(alias), alias[9999])


main()
