class Scale(object):
    def __init__(self, k):
        self.k = k

    def __call__(self, u):
        return self.k * u


class Poly(object):
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __call__(self, x):
        total = 0
        for c in self.coeffs:
            total = total * x + c
        return total


def main():
    s = Scale(3)
    print(s(2), s(2.5))
    fs = [Scale(1), Scale(2), Poly([1, 0, -1])]
    print([f(3) for f in fs])
    print(fs[2](2), fs[0](10))
    p = Poly([2, 3])
    total = 0
    for i in range(5):
        total += p(i)
    print(total)


main()
