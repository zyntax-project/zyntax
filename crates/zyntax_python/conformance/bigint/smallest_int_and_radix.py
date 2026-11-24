# zypy: bigint
# The smallest 64-bit int under negation and floor division, the text of
# big ints in base 16, 8 and 2, and the hash of ints past 2 ** 61 - 1.

m = -(1 << 63)
print(m, -m, -(-m), m // -1, m // 1, m // 2, m // -2, (m + 1) // -1)
print(7 // 2, -7 // 2, 7 // -2, -7 // -2, 0 // 5)
n = 9
n //= 2
print(n, -n, -(-n))
q = m
q //= -1
print(q, isinstance(q, int))
print([-x for x in (m, 1, -1, 0)], [x // -1 for x in (m, 5, -5)])

big = (1 << 200) + 0xabcdef0123456789
print(hex(big), oct(big), bin(big)[:40], len(bin(big)))
print(hex(-big), oct(-big)[:12], bin(-big)[:12])
print(hex(255), oct(8), bin(5), hex(-255), hex(1 << 62), hex(m))
print(hex(1 << 60), hex((1 << 60) - 1), hex(1 << 120), oct(1 << 60), bin(1 << 60))
print(int(hex(big), 16) == big, int(oct(big), 8) == big, int(bin(big), 2) == big)
print(format(255, "x"), format(255, "#06x"), format(-255, "b"), format(3.14159, ".2f"), format("ab", ">4") + "|", format(7))

for x in (1 << 61, (1 << 61) - 1, (1 << 61) - 2, (1 << 62) + 1, (1 << 63) - 1, m, m + 1, -1, -2, -((1 << 61) - 1), -(1 << 61), 12345, -12345):
    print(hash(x), end=" ")
print()
print(hash(1 << 70), hash(-(1 << 70)), hash(3), hash(True), hash(0))
