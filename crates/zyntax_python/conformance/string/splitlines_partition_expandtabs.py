for t in ["a\nb\r\nc\rd", "a\n\nb", "\n", "", "abc", "a\r\n", "x\ny\n", "\r\n\r\n"]:
    print(repr(t), t.splitlines(), t.splitlines(True), t.splitlines(False))
for t in ["a,b,c", "abc", "", "a,b,,c", "ünï,x"]:
    for sep in [",", ",,", "z", "abc", "ï"]:
        print(repr(t), repr(sep), t.partition(sep), t.rpartition(sep))
try:
    "abc".partition("")
except ValueError as e:
    print("ValueError", e)
try:
    "abc".rpartition("")
except ValueError as e:
    print("ValueError", e)
head, sep, tail = "key=value=more".partition("=")
print(head, sep, tail)
head, sep, tail = "key=value=more".rpartition("=")
print(head, sep, tail)
for t in ["a\tb", "ab\tc\td", "\t", "a\nb\tc", "abcdefgh\ti", "ü\tb"]:
    print(repr(t.expandtabs()), repr(t.expandtabs(0)), repr(t.expandtabs(1)), repr(t.expandtabs(3)), repr(t.expandtabs(4)), repr(t.expandtabs(-2)))
s = "hello world"
print(s.startswith(("he", "xx")), s.endswith(("he", "xx")))
print(s.startswith(("xx", "ld")), s.endswith(("xx", "ld")))
print(s.startswith(("xx", "yy")), s.endswith(("xx", "yy")))
print(s.startswith(("hello world",)), s.endswith(("hello world",)))
print(s.startswith(("",)), s.endswith(("",)))
