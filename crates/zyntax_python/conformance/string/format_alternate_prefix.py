n = 255
neg = -255
print(f"{n:#06x}|{n:#x}|{n:#X}|{n:#o}|{n:#b}|{n:#010b}")
print(f"{neg:#06x}|{neg:#x}|{neg:+#x}|{n:+#06x}|{n: #06x}")
print(f"{n:>#8x}|{n:<#8x}|{n:^#8x}|{n:*=#8x}|{n:0=#8x}")
print(f"{n:#_x}|{n:#06_x}|{n:+08d}")
print("{:#06x}".format(n), "%#06x" % n, "%#x" % neg)
x = 3.5
print(f"{x:08.2f}|{-x:08.2f}|{x:+08.2f}|{x:=8.2f}")
v = [n, neg]
print([f"{i:#06x}" for i in v])
