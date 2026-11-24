s = "a\x0bb\x0cc\x1cd\x1de\x1ef\x85g h i"
print(s.splitlines())
print(s.splitlines(True))
print("a\r\nb\rc\nd\n".splitlines(), "a\r\nb\rc\nd\n".splitlines(keepends=True))
print("".splitlines(), "\n".splitlines(), "\n\n".splitlines(), "x".splitlines(), "x\n\ny".splitlines())
print("é\nü\r\nñ".splitlines(True))
print("a\n\r\nb".splitlines(), "\r\n".splitlines(True), "\r\r\n".splitlines(True))
text = "one\ntwo\n"
for line in text.splitlines():
    print(len(line), line)
