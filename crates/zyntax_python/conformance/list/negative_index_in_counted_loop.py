# An index below the counter's first value reads from the end of the
# list, as CPython's does, in a loop over the whole list.
from array import array


def previous(xs: list[int]) -> list[int]:
    s: list[int] = []
    for i in range(len(xs)):
        s.append(xs[i - 1])
    return s


def previous_f(xs: array) -> list[float]:
    s: list[float] = []
    for i in range(len(xs)):
        s.append(xs[i - 1])
    return s


def from_end(xs: list[int], n: int) -> int:
    t = 0
    for i in range(n):
        t = t * 10 + xs[i - n]
    return t


def main() -> None:
    for rep in range(300):
        a = previous([1, 2, 3])
        b = previous_f(array("d", [1.0, 2.0, 3.0]))
        c = from_end([1, 2, 3, 4], 3)
        d = from_end([1, 2, 3, 4], 4)
        if rep == 0 or rep == 299:
            print(a, b, c, d)
        try:
            from_end([1, 2], 3)
        except IndexError as e:
            if rep == 0:
                print("IndexError:", e)


main()
