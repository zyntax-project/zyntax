# zypy: bigint
class Counter:
    def __init__(self):
        self.n = 1

    def bump(self):
        self.n *= 10 ** 10

    def total(self, k):
        return self.n + k


c = Counter()
for _ in range(4):
    c.bump()
print(c.n, c.total(5))


def power_sum(base, count):
    acc = 0
    term = 1
    for _ in range(count):
        acc += term
        term *= base
    return acc


print(power_sum(10, 30), power_sum(2, 100), power_sum(3, 5))


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


print(gcd(2 ** 100, 6 ** 50), gcd(12345678901234567890, 9876543210))


def digits(n):
    out = []
    while n > 0:
        n, r = divmod(n, 10)
        out.append(r)
    return out[::-1]


print(digits(2 ** 100))
print(sum(digits(2 ** 1000)))
values = [i ** 20 for i in range(1, 6)]
print(values, sum(values), max(values) - min(values))
print(str(2 ** 64) == "18446744073709551616", 2 ** 64 == 18446744073709551616)
