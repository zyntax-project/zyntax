# Floor division and modulo by a literal power of two, on both signs:
# the same answers as by any other divisor.
for a in [0, 1, 7, -1, -7, -8, 1023, -1024]:
    print(a, a // 1, a % 1, a // 2, a % 2, a // 8, a % 8, a // 1024, a % 1024)

def halve(n: int) -> int:
    return n // 2

def low_bits(n: int) -> int:
    return n % 16

print(halve(-9), halve(9), low_bits(-1), low_bits(33))
x = -5
x //= 4
y = -5
y %= 4
print(x, y)
