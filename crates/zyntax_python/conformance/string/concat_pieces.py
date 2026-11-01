# Chains of concatenation and formatting that join many pieces at once:
# the result is the pieces in order, text or bytes.
a, b, c = "x", "yé", "z"
n = 7
print(a + b + c)
print(a + b + c + a + b + c + a + b + c + a)
print(a + (b + c) + (a + b))
print(f"{a}-{b}-{c}-{n}-{n * 2}")
print("%s|%s|%d|%5.2f" % (a, b, n, 3.14159))
print("{}:{}:{}".format(a, b, n))
print(a, b, c, n, sep="--", end="!\n")

s = ""
for i in range(5):
    s = s + str(i) + ","
print(s)
print(repr("" + "" + ""), len("é" + "é" + "é"))

x = b"ab" + b"cd" + b"\xc3" + b"\xa9"
print(x, len(x))
print(b"%c%c%c" % (65, 66, 67), b"%d-%s" % (5, b"zz"))
print(b"\xff" + b"a" + b"b")
print(b"\xff" + b"\xff" + b"\xff", len(b"\xff" + b"\xff" + b"\xff"))
print((b"\xc3" + b"\xa9" + b"!").decode("utf-8"))

pixels = b""
for v in (0, 255, 7):
    pixels = pixels + b"%c%c%c" % (v, v, v)
print(pixels, len(pixels))

many = "a" + "b" + "c" + "d" + "e" + "f" + "g" + "h" + "i" + "j" + "k" + "l"
print(many, len(many))
