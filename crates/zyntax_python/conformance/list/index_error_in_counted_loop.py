# A counted loop whose index leaves the list raises where CPython does,
# after the same writes; one that stays inside finishes.


def bump(xs: list[int], n: int) -> int:
    done = 0
    for i in range(n):
        xs[i] += 1
        done += 1
    return done


def axpy(a: float, xs: list[float], ys: list[float], n: int) -> None:
    for i in range(n):
        ys[i] = a * xs[i] + ys[i]


def ahead(xs: list[int], lo: int, hi: int) -> int:
    t = 0
    for i in range(lo, hi):
        t += xs[i + 1] - xs[i]
    return t


def popping(xs: list[int], n: int) -> int:
    t = 0
    for i in range(n):
        t += xs[i]
        xs.pop()
    return t


def deleting(xs: list[int], n: int) -> int:
    t = 0
    for i in range(n):
        t += xs[i]
        del xs[0]
    return t


def shrink(xs: list[int]) -> None:
    xs.pop()


def calling(xs: list[int], n: int) -> int:
    t = 0
    for i in range(n):
        t += xs[i]
        shrink(xs)
    return t


def main() -> None:
    for rep in range(300):
        show = rep == 0 or rep == 299
        xs = [0] * 16
        try:
            bump(xs, 20)
        except IndexError as e:
            if show:
                print("IndexError:", e, sum(xs))
        done = bump(xs, 16)
        if show:
            print(done, sum(xs), bump(xs, 0), bump(xs, -2))
        xf = [float(i) for i in range(8)]
        yf = [1.0] * 8
        axpy(2.0, xf, yf, 8)
        try:
            axpy(2.0, xf, yf, 9)
        except IndexError as e:
            if show:
                print("IndexError:", e)
        if show:
            print(yf)
        sq = [i * i for i in range(10)]
        if show:
            print(ahead(sq, 0, 9), ahead(sq, 3, 3), ahead(sq, -3, 2))
        try:
            ahead(sq, 2, 10)
        except IndexError as e:
            if show:
                print("IndexError:", e)
        for name, f in (("popping", popping), ("deleting", deleting), ("calling", calling)):
            zs = list(range(10))
            try:
                r = f(zs, 8)
                if show:
                    print(name, r, zs)
            except IndexError as e:
                if show:
                    print(name, "IndexError:", e, zs)


main()
