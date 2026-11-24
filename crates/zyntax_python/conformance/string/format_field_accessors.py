class P:
    def __init__(self):
        self.x = 4
        self.tag = "t"


p = P()
pair = [7, 8]
d = {"k": 5}
print("{1[0]} {0[1]}".format(pair, pair))
print("{0.x}-{0.tag}".format(p))
print("{p.x} {pair[1]} {d[k]}".format(p=p, pair=pair, d=d))
print("{0[1]:>4}|{1.x:03d}".format(pair, p))
print("{.x}".format(p))
