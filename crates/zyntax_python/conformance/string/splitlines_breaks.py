# Line breaks of every kind, in runs, at the ends and mixed.
cases = ["", "a", "\n", "\r", "\r\n", "a\nb", "a\rb", "a\r\nb", "a\n\rb", "\n\n",
         "\r\r", "\r\n\r\n", "a\r\n", "a\rb\nc\r\nd\n", "x\ny\rz\r\nw", "tail\r",
         "\ra\n", "a\r\r\nb"]
for c in cases:
    print([ord(x) for x in c], c.splitlines())

big = "line of text\r\n" * 3000 + "mixed\r" * 3000 + "end\n" * 3000
lines = big.splitlines()
print(len(lines), lines[0], lines[2999], lines[3000], lines[5999], lines[6000], lines[-1])
print(len(("line of text\n" * 40000).splitlines()))
