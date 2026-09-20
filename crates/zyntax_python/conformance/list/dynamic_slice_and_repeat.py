from array import array


def repeat(xs, n):
    return xs * n


def fill(target, source):
    target[:] = source[:]


def main():
    print(repeat([1, 2], 3), repeat("ab", 2), repeat((1,), 2))
    holders = [[0.0] * 3, array('d', [0]) * 3, [None] * 3]
    fill(holders[0], [1.5, 2.5, 3.5])
    fill(holders[1], array('d', [4, 5, 6]))
    fill(holders[2], ["a", "b", "c"])
    print(holders)
    xs = [1, 2, 3, 4]
    fill(xs, range(4, 8))
    print(xs)
    print(int("42"), int(" -7 "), float("2.5"), float("3"))
    for bad in ["x", "1.5", ""]:
        try:
            int(bad)
        except ValueError:
            print("ValueError", repr(bad))
    try:
        float("abc")
    except ValueError:
        print("ValueError float")


main()
