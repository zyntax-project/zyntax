# Unpacking a tuple whose items are computed by statements first.
for a, b, st in [(10, 0, -1), (2**40, 2**40 + 3, 1)]:
    print(a, b, st)
import struct
n, = struct.unpack("<I", b"\x05\x00\x00\x00")
print(n)
pairs = [(2**3, 2**4), (3**2, 5)]
for a, b in pairs:
    print(a + b)
x, y = (2**10, 7)
print(x, y)
