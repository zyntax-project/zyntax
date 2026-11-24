# zypy: bigint
# Values of thousands of bits: the product of long operands, division by
# long and short divisors, shifts across many limbs, and text round trips.

def factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

f = factorial(700)
print(len(str(f)), f % 1000003, f // factorial(690) % 10007)
print(str(f)[:30], str(f)[-30:])

a = 3 ** 4000
b = 7 ** 2500
p = a * b
print(len(str(p)), p % 998244353, (p // a) == b, p % a, p % b)
print(divmod(p + 12345, a)[1], divmod(p + 12345, b)[1])
q, r = divmod(a, b)
print(len(str(q)), len(str(r)), q * b + r == a)

big = (1 << 5000) - 1
print(big.bit_length(), (big >> 4990), (big << 20) >> 20 == big)
print(int(str(big)) == big, str(big)[:20])

square = (10 ** 600 + 7) ** 2
print(square % 10 ** 9, square // 10 ** 590 % 100000)
print((a - b) + (b - a), (a - b) * (b - a) < 0, -a // b, -a % b)

x = factorial(300)
y = x * x
print(y // x == x, y % x, (y + 1) % x, (y - 1) % x == x - 1)
print(sum(int(c) for c in str(factorial(200))))
