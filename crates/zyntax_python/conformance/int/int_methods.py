# The methods and attributes of int, bool and float values.

x = 255
print(x.bit_length(), (0).bit_length(), (-5).bit_length(), (1 << 40).bit_length())
print(x.bit_count(), (-7).bit_count(), True.bit_count())
print(x.real, x.imag, x.numerator, x.denominator, x.conjugate())
print(x.as_integer_ratio(), x.is_integer())
print(True.real, True.denominator)

print((258).to_bytes(2, "big"), (258).to_bytes(2, "little"), (258).to_bytes(length=3, byteorder="big"))
print((255).to_bytes(), (-1).to_bytes(2, "big", signed=True), (-2).to_bytes(3, byteorder="little", signed=True))
print(int.from_bytes(b"\x01\x02", "big"), int.from_bytes(b"\x01\x02", "little"))
print(int.from_bytes(b"\xff\xff", "big", signed=True), int.from_bytes(b"\xff\xff", byteorder="big"))
print(int.from_bytes(b"", "big"), int.from_bytes(bytes([7])))

size = 1000
print(int.from_bytes(size.to_bytes(4, "big"), "big") == size)

f = 2.5
print(f.is_integer(), (3.0).is_integer(), float("inf").is_integer(), f.real, f.imag, f.conjugate())

for bad in (lambda: (256).to_bytes(1, "big"), lambda: (-1).to_bytes(2, "big"), lambda: (1).to_bytes(2, "middle")):
    try:
        bad()
    except (OverflowError, ValueError) as e:
        print(type(e).__name__, e)
