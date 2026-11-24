b = bytearray(b"hello")
b[0] = 72
b.append(33)
b.extend(b" ok")
print(b, len(b), b[0], b[-1], bytes(b), b.decode(), list(b)[:3])
print(bytearray(), bytearray(3), bytearray([104, 105]), bytearray("hi", "utf-8"), bytearray(b"ab") + b"c", bytearray(b"ab") * 2)
c = bytearray(b"abc")
print(c.pop(), c, c.pop(0), c, c == bytearray(b"b"), c == b"b", c != b"x", 98 in c, 1 in c)
c.insert(0, 120)
c.reverse()
d = c.copy()
c.clear()
print(c, d, isinstance(d, bytearray), isinstance(b"x", bytearray))
total = 0
for byte in bytearray(b"\x01\x02\x03"):
    total += byte
buffer = bytearray()
for i in range(5):
    buffer += bytes([65 + i])
print(total, buffer, buffer.decode())
