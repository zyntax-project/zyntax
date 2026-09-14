# Operators on instances: through the class's method where the operand
# types are known, and through the same method when the operands are
# dynamic. An operator the class does not define is a TypeError.

class Vec:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __mul__(self, k):
        return Vec(self.x * k, self.y * k)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)

def main():
    a = Vec(1.0, 2.0)
    b = Vec(10.0, 20.0)
    c = a + b
    print(c.x, c.y)
    d = c * 3
    print(d.x, d.y)
    acc = Vec(0.0, 0.0)
    for i in range(4):
        acc = acc + a
        acc += b
    print(acc.x, acc.y)
    e = (a - b) * 2
    print(e.x, e.y)
    xs = [a, 5]
    f = xs[0] + b
    print(f.x, f.y)
    g = xs[0] * xs[1]
    print(g.x, g.y)
    try:
        print(xs[1] + xs[0])
    except TypeError as err:
        print("TypeError")
    try:
        print(xs[0] // xs[0])
    except TypeError as err:
        print("TypeError")

main()
