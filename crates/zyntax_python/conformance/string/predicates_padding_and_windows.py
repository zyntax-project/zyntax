words = ["", "abc", "ABC", "aBc", "Hello World", "hello world", "123", "12a", " \t\n", "  x ", "ünï", "ÜNÏ", "a_b1", "_x", "1x", "x y", "Hello", "HELLO", "straße", "ǅ"]
for w in words:
    print(repr(w), w.isalpha(), w.isalnum(), w.isdigit(), w.isdecimal(), w.isnumeric(), w.isspace(),
          w.isupper(), w.islower(), w.istitle(), w.isascii(), w.isidentifier())
for w in words:
    print(repr(w), repr(w.swapcase()), repr(w.casefold()))
for w in ["", "ab", "abc", "abcd", "-12", "+5", "ünï"]:
    for n in [-1, 0, 1, 2, 3, 6, 7]:
        print(repr(w), n, repr(w.ljust(n)), repr(w.rjust(n)), repr(w.center(n)), repr(w.zfill(n)))
    print(repr(w.ljust(6, "*")), repr(w.rjust(6, "*")), repr(w.center(6, "*")), repr(w.center(7, "*")))
s = "abcabcabc"
u = "äbcäbcäbc"
print(repr(s), '"b"', s.find("b"), s.rfind("b"), s.count("b"))
print(repr(s), '"b", 2', s.find("b", 2), s.rfind("b", 2), s.count("b", 2))
print(repr(s), '"b", 2, 8', s.find("b", 2, 8), s.rfind("b", 2, 8), s.count("b", 2, 8))
print(repr(s), '"b", -3', s.find("b", -3), s.rfind("b", -3), s.count("b", -3))
print(repr(s), '"b", -3, -1', s.find("b", -3, -1), s.rfind("b", -3, -1), s.count("b", -3, -1))
print(repr(s), '"b", 20', s.find("b", 20), s.rfind("b", 20), s.count("b", 20))
print(repr(s), '"b", 0, 1', s.find("b", 0, 1), s.rfind("b", 0, 1), s.count("b", 0, 1))
print(repr(s), '"b", 5, 3', s.find("b", 5, 3), s.rfind("b", 5, 3), s.count("b", 5, 3))
print(repr(s), '"", 0', s.find("", 0), s.rfind("", 0), s.count("", 0))
print(repr(s), '"", 3', s.find("", 3), s.rfind("", 3), s.count("", 3))
print(repr(s), '"", 9', s.find("", 9), s.rfind("", 9), s.count("", 9))
print(repr(s), '"", 10', s.find("", 10), s.rfind("", 10), s.count("", 10))
print(repr(s), '"", 2, 1', s.find("", 2, 1), s.rfind("", 2, 1), s.count("", 2, 1))
print(repr(s), '"bc", 1', s.find("bc", 1), s.rfind("bc", 1), s.count("bc", 1))
print(repr(s), '"bca", 0, 5', s.find("bca", 0, 5), s.rfind("bca", 0, 5), s.count("bca", 0, 5))
print(repr(s), '"a", 0, 100', s.find("a", 0, 100), s.rfind("a", 0, 100), s.count("a", 0, 100))
print(repr(s), '"b", 0, None', s.find("b", 0, None), s.rfind("b", 0, None), s.count("b", 0, None))
print(repr(s), '"b", None, 5', s.find("b", None, 5), s.rfind("b", None, 5), s.count("b", None, 5))
print(repr(u), '"b"', u.find("b"), u.rfind("b"), u.count("b"))
print(repr(u), '"b", 2', u.find("b", 2), u.rfind("b", 2), u.count("b", 2))
print(repr(u), '"b", 2, 8', u.find("b", 2, 8), u.rfind("b", 2, 8), u.count("b", 2, 8))
print(repr(u), '"b", -3', u.find("b", -3), u.rfind("b", -3), u.count("b", -3))
print(repr(u), '"b", -3, -1', u.find("b", -3, -1), u.rfind("b", -3, -1), u.count("b", -3, -1))
print(repr(u), '"b", 20', u.find("b", 20), u.rfind("b", 20), u.count("b", 20))
print(repr(u), '"b", 0, 1', u.find("b", 0, 1), u.rfind("b", 0, 1), u.count("b", 0, 1))
print(repr(u), '"b", 5, 3', u.find("b", 5, 3), u.rfind("b", 5, 3), u.count("b", 5, 3))
print(repr(u), '"", 0', u.find("", 0), u.rfind("", 0), u.count("", 0))
print(repr(u), '"", 3', u.find("", 3), u.rfind("", 3), u.count("", 3))
print(repr(u), '"", 9', u.find("", 9), u.rfind("", 9), u.count("", 9))
print(repr(u), '"", 10', u.find("", 10), u.rfind("", 10), u.count("", 10))
print(repr(u), '"", 2, 1', u.find("", 2, 1), u.rfind("", 2, 1), u.count("", 2, 1))
print(repr(u), '"bc", 1', u.find("bc", 1), u.rfind("bc", 1), u.count("bc", 1))
print(repr(u), '"bca", 0, 5', u.find("bca", 0, 5), u.rfind("bca", 0, 5), u.count("bca", 0, 5))
print(repr(u), '"a", 0, 100', u.find("a", 0, 100), u.rfind("a", 0, 100), u.count("a", 0, 100))
print(repr(u), '"b", 0, None', u.find("b", 0, None), u.rfind("b", 0, None), u.count("b", 0, None))
print(repr(u), '"b", None, 5', u.find("b", None, 5), u.rfind("b", None, 5), u.count("b", None, 5))
try:
    print("index", '"b"', s.index("b"))
except ValueError as e:
    print("index", '"b"', "ValueError", e)
try:
    print("rindex", '"b"', s.rindex("b"))
except ValueError as e:
    print("rindex", '"b"', "ValueError", e)
try:
    print("index", '"b", 2', s.index("b", 2))
except ValueError as e:
    print("index", '"b", 2', "ValueError", e)
try:
    print("rindex", '"b", 2', s.rindex("b", 2))
except ValueError as e:
    print("rindex", '"b", 2', "ValueError", e)
try:
    print("index", '"z"', s.index("z"))
except ValueError as e:
    print("index", '"z"', "ValueError", e)
try:
    print("rindex", '"z"', s.rindex("z"))
except ValueError as e:
    print("rindex", '"z"', "ValueError", e)
try:
    print("index", '"b", 5, 9', s.index("b", 5, 9))
except ValueError as e:
    print("index", '"b", 5, 9', "ValueError", e)
try:
    print("rindex", '"b", 5, 9', s.rindex("b", 5, 9))
except ValueError as e:
    print("rindex", '"b", 5, 9', "ValueError", e)
try:
    print("index", '"a", 1', s.index("a", 1))
except ValueError as e:
    print("index", '"a", 1', "ValueError", e)
try:
    print("rindex", '"a", 1', s.rindex("a", 1))
except ValueError as e:
    print("rindex", '"a", 1', "ValueError", e)
print('"a"', s.startswith("a"), s.endswith("a"))
print('"a", 0', s.startswith("a", 0), s.endswith("a", 0))
print('"b", 1', s.startswith("b", 1), s.endswith("b", 1))
print('"b", 1, 2', s.startswith("b", 1, 2), s.endswith("b", 1, 2))
print('"bc", 1, 3', s.startswith("bc", 1, 3), s.endswith("bc", 1, 3))
print('"c", -1', s.startswith("c", -1), s.endswith("c", -1))
print('"c", 0, -1', s.startswith("c", 0, -1), s.endswith("c", 0, -1))
print('"", 9', s.startswith("", 9), s.endswith("", 9))
print('"", 10', s.startswith("", 10), s.endswith("", 10))
print('"abc", 3', s.startswith("abc", 3), s.endswith("abc", 3))
print('"abc", 0, 2', s.startswith("abc", 0, 2), s.endswith("abc", 0, 2))
print('"", 0, 0', s.startswith("", 0, 0), s.endswith("", 0, 0))
print(s.replace("a", "X", 0), s.replace("a", "X", 1), s.replace("a", "X", 2), s.replace("a", "X", -1), s.replace("a", "X", 9))
print("abc".replace("", "-", 2), "abc".replace("", "-"), "abc".replace("", "-", 0))
print(s.removeprefix("abc"), s.removeprefix("x"), s.removeprefix(""), s.removesuffix("abc"), s.removesuffix("x"), s.removesuffix(""), "abc".removeprefix("abc"), "abc".removesuffix("abc"))
