# zypy: bigint
print(2 ** 100)
print(10 ** 20 // 3)
print((2 ** 64) % 1000007)
x = 9223372036854775807
print(x + 1, x * 2, -x - 2, x - (-x), (-x - 1) - 1)
print(12345678901234567890123 * 3, -12345678901234567890123)
print(2 ** 63 - 1 + 1, -(2 ** 63) - 1, (2 ** 63) - 1, 9223372036854775807 + 0)
print(0 ** 0, 5 ** 0, (-2) ** 63, (-2) ** 64, 2 ** -2, (-3) ** 3)
print(1 << 62, 1 << 63, 1 << 64, (1 << 64) >> 64, 7 << 0)


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


print(fib(100), fib(300))
print(fact(30), fact(25) // fact(23))
big = 3 ** 150
small = 12
print(big > small, small < big, big == big, big != small, big >= big, small > big)
print(big // small, big % small, divmod(big, 1000), divmod(-big, 7))
print(-big, abs(-big), -big // 7, -big % 7, big // -7, big % -7)
print(big << 10, big >> 100, (-big) >> 3, big & 0xFFFFFFFFFFFF, big | 1, big ^ (big >> 5), ~big)
print(big / 3, (3 ** 400) / (3 ** 399), 1 / big, float(big), float(2 ** 70))
print(big * big * big == big ** 3, (big + 1) * (big - 1) == big * big - 1)
total = 0
for i in range(1, 70):
    total += i * 10 ** 15
print(total)
