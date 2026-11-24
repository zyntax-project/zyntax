# The search, edit, case, class and split methods of bytes.

data = b"hello, world, hello"
print(data.find(b"hello"), data.find(b"hello", 1), data.rfind(b"hello"), data.find(b"zzz"))
print(data.find(b"o", 5, 9), data.find(b"o", -3), data.find(111), data.find(b""))
print(data.count(b"hello"), data.count(b"l"), data.count(b""), data.count(b"l", 0, 4))
print(data.index(b"world"), data.rindex(b"hello"))
print(data.startswith(b"hello"), data.startswith(b"world"), data.startswith(b"world", 7))
print(data.endswith(b"hello"), data.endswith(b"lo", 0, 5), data.endswith((b"x", b"llo")))
print(data.startswith((b"x", b"y")), data.startswith((b"he", b"x")))

print(data.replace(b"hello", b"bye"))
print(data.replace(b"l", b"L", 2))
print(b"abc".replace(b"", b"-"))
print(b"abc".replace(b"", b"-", 2))

print(b"Hello World".upper(), b"Hello World".lower(), b"Hello World".swapcase())
print(b"hELLO wORLD".capitalize(), b"hello wORLD-x 1a".title())
print(b"caf\xc3\xa9".upper(), b"caf\xc3\xa9".title())

print(b"abc".isalpha(), b"abc1".isalpha(), b"abc1".isalnum(), b"123".isdigit(), b"".isdigit())
print(b" \t\n".isspace(), b"ABC".isupper(), b"abc".islower(), b"Hello World".istitle())
print(b"abc".isascii(), b"caf\xc3\xa9".isascii(), b"".isascii())

print(b"  padded \n".strip(), b"  padded \n".lstrip(), b"  padded \n".rstrip())
print(b"xxabcxx".strip(b"x"), b"xyabcyx".strip(b"xy"), b"xxabcxx".lstrip(b"x"), b"xxabcxx".rstrip(b"x"))
print(b"  x ".strip(None))

print(b"ab".ljust(5), b"ab".rjust(5, b"."), b"ab".center(6, b"*"), b"ab".center(5, b"*"))
print(b"-42".zfill(6), b"7".zfill(3), b"abc".ljust(2))

print(b"a,b,,c".split(b","), b"a,b,c".split(b",", 1), b"a,b,c".rsplit(b",", 1))
print(b"  a  b c ".split(), b"  a  b c ".split(None, 1), b"  a  b c ".rsplit(None, 1))
print(b"a--b--c".split(b"--"), b"".split(b","), b"".split())
print(b"line1\nline2\r\nline3\rline4".splitlines(), b"a\nb\n".splitlines(True))
print(b"key=value=x".partition(b"="), b"key=value=x".rpartition(b"="))
print(b"novalue".partition(b"="), b"novalue".rpartition(b"="))
print(b"a b c".split(sep=b" ", maxsplit=1), b"x\ny".splitlines(keepends=True))

print(b", ".join([b"a", b"b", b"c"]), b"".join([b"x", b"y"]), b"-".join([]))
print(b"-".join((b"p", b"q")), b" ".join(b"ab cd".split()))
print(b"prefix-body".removeprefix(b"prefix-"), b"body.txt".removesuffix(b".txt"), b"body".removeprefix(b"zzz"))

parts = b"one two three".split()
print(len(parts), parts[1], parts[1].upper())
for p in b"a:b".split(b":"):
    print(p.decode(), len(p))

print(b"a\tbc\td".expandtabs(4), b"a\tb".expandtabs())
table = bytes.maketrans(b"abc", b"xyz")
print(len(table), b"aabbccdd".translate(table), b"aabbccdd".translate(table, b"d"))
print(b"hello".translate(None, b"l"), bytes.fromhex("48 65 6c6C6f"))

# Bytes that are not text.
raw = b"\x80\x80ab\xff\x00\x80"
print(raw.find(b"\x80", 1), raw.count(b"\x80"), raw.split(b"\x80"), raw.replace(b"\x80", b"."))
print(raw.upper(), raw.isascii(), raw.partition(b"ab"))
split_mid = b"\xc3\xa9".split(b"\xa9")
print(split_mid)

# Errors speak Python.
for bad in (lambda: b"abc".index(b"z"), lambda: b"abc".rindex(b"z"), lambda: b"a".split(b"")):
    try:
        bad()
    except ValueError as e:
        print("ValueError", e)
try:
    b"-".join([b"a", 1])
except TypeError as e:
    print("TypeError", e)
