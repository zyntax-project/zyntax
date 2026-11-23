texts = ["a,b,c", ",a,,b,", "", "abc", "a--b----c", "aaa", "ünï,çödé,x", "a b  c", "  a  b  c  ", "   ", "a\tb\nc"]
for t in texts:
    print(repr(t), t.split(), t.rsplit(), t.split(None), t.rsplit(None))
    print(t.split(","), t.rsplit(","), t.split("--"), t.rsplit("--"), t.split("aa"), t.rsplit("aa"), t.split("ü"), t.rsplit("ü"))
    print(t.split(",", 1), t.rsplit(",", 1), t.split(",", 0), t.rsplit(",", 0), t.split(",", -1), t.rsplit(",", -1), t.split(",", 5), t.rsplit(",", 5))
    print(t.split("aa", 1), t.rsplit("aa", 1), t.split(" ", 1), t.rsplit(" ", 1), t.split("abc", 1), t.rsplit("abc", 1))
    print(t.split(None, 1), t.rsplit(None, 1), t.split(None, 0), t.rsplit(None, 0), t.split(None, 2), t.rsplit(None, 2), t.split(None, -1), t.rsplit(None, -1))
try:
    "abc".split("")
except ValueError as e:
    print("ValueError", e)
try:
    "abc".rsplit("", 1)
except ValueError as e:
    print("ValueError", e)
for t in ["x\x0by\x1cz", "a\u00a0b\u2003c", " \x85a\u2028b\u3000"]:
    print(len(t.split()), len(t.rsplit()), len(t.split(None, 1)), len(t.rsplit(None, 1)))
