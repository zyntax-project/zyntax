# An int against a float compares exactly, beyond 2^53 too.
def cmp(a, b):
    print(a < b, a <= b, a > b, a >= b, a == b, a != b)


def typed(i: int, f: float):
    print(i < f, i <= f, i > f, i >= f, i == f, i != f)
    print(f < i, f <= i, f > i, f >= i, f == i, f != i)


big = 2**53 + 1
print(2**53 + 1 > 2.0**53, big > 2.0**53, big == 2.0**53)
typed(2**53 + 1, 2.0**53)
typed(2**53, 2.0**53)
typed(2**60, 2.0**60 + 1.0)
typed(2**63 - 1, 2.0**63)
typed(-(2**63), -(2.0**63))
typed(5, float("nan"))
typed(5, float("inf"))
typed(-5, float("-inf"))
typed(3, 3.5)
typed(-3, -3.5)
typed(7, 7.0)
typed(2**62 + 1, 1e300)
typed(2**62 + 1, -1e300)
cmp(2**53 + 1, 2.0**53)
cmp(2.0**53, 2**53 + 1)
x = 2**53
y = 2**53 + 1
f = 2.0**53
print(x == f, y == f, y > f, f < y, 0.5 < y, y < 1e20)
for k in range(-1, 3):
    i = x + k
    print(i, i < f, i == f, i > f, f <= i)
print(1 < 2.5, 3 == 3.0, True < 1.5, 2.5 > False)
