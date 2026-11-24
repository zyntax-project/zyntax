import re

print(re.findall(r"\d+", "a1b22c333"), re.findall(r"[a-z]+", "Hello World foo"), re.findall(r"\w+@\w+\.com", "me@x.com you@y.com bad@z.org"))
print(re.findall(r"(\d+)-(\d+)", "1-2 33-44"), re.findall(r"(\w)(\d)?", "a1b"), re.findall(r"", "ab"))
print(re.sub(r"\s+", " ", "a   b \t c"), re.sub(r"(\w+) (\w+)", r"\2 \1", "hello world"), re.sub(r"\d", "#", "a1b2", count=1))
print(re.sub(r"(?P<n>\d+)", lambda m: str(int(m.group("n")) * 2), "a1 b22"), re.sub("x*", "-", "abc"))
print(re.subn(r"a", "b", "banana"), re.split(r"\s*,\s*", "a , b,c ,d"), re.split(r"(-)", "1-2-3"), re.split(r"\d", "a1b2c", maxsplit=1))
m = re.match(r"(?P<year>\d{4})-(?P<month>\d\d)", "2024-05-17 extra")
print(m.group(), m.group(1), m.group("month"), m.groups(), m.groupdict(), m.span(), m.start(2), m.end("year"), m[1])
print(re.match(r"a", "ba"), re.search(r"a", "ba").span(), re.fullmatch(r"\d+", "123") is not None, re.fullmatch(r"\d+", "123x"))
print(bool(re.match(r"a.c", "abc")), bool(re.match(r"a.c", "a\nc")), bool(re.match(r"a.c", "a\nc", re.S)), bool(re.match("ABC", "abc", re.I)))
print(re.findall(r"^\w+", "one\ntwo\nthree", re.M), re.findall(r"\w+$", "one\ntwo\nthree", re.M), re.findall(r"^\w+", "one\ntwo"))
print(re.findall(r"a*", "baaac"), re.findall(r"a+?", "aaa"), re.findall(r"<.*?>", "<a><b>"), re.findall(r"<.*>", "<a><b>"))
print(re.findall(r"a{2}", "aaaaa"), re.findall(r"a{2,3}", "aaaaaaa"), re.findall(r"a{,2}", "aaa"), re.findall(r"x{1,2}?", "xxx"))
print(re.findall(r"\bcat\b", "cat concat cat."), re.findall(r"\Bat", "cat at"), re.findall(r"[^aeiou\s]+", "hello world"))
print(re.findall(r"(a|b)+", "abab"), re.findall(r"(?:a|b)+", "abab"), re.findall(r"cat|dog|bird", "dog cat bird fish"))
print(re.findall(r"(\w)\1", "aabbcd"), re.findall(r"(?P<c>\w)(?P=c)", "aabbcd"), re.match(r"(\d+)\s\1", "12 12") is not None)
print(re.findall(r"\w+(?= is)", "this is it"), re.findall(r"\d+(?!\d|px)", "10px 20em 30"), re.findall(r"[\d.]+", "v1.2.3 and 4.5"))
print(re.findall(r"[a-c-]+", "a-b-c d"), re.findall(r"[\]\[]", "[x]"), re.findall(r"\$\d+\.\d\d", "cost $12.50 or $3.99"))
print(re.escape("a.b*c"), re.findall(re.escape("1+1"), "1+1=2"), re.compile(r"(\d)(\d)").groups, re.compile(r"(?P<a>.)").groupindex)
pattern = re.compile(r"(\w+)@(\w+)\.com")
print(pattern.findall("a@b.com c@d.com"), pattern.search("zz a@b.com").group(2), [m.group(1) for m in pattern.finditer("a@b.com c@d.com")])
print(re.search(r"(a)|(b)", "b").groups(), re.search(r"(a)?b", "b").group(1), re.search(r"(a)?b", "b").groups(""))
print(re.findall(r"""
    \d+   # digits
    \s*   # space
""", "12 34", re.X), re.search(r"(?i)HELLO", "say hello").group())
for bad in [r"(", r"[a", r"*a", r"a)", r"\q"]:
    try:
        re.compile(bad)
    except re.error:
        print("error")
words = re.findall(r"[A-Za-z']+", "Don't stop, can't stop!")
print(words, len(words), re.sub(r"[aeiou]", "", "education"), re.sub(r"^\s+|\s+$", "", "  trim me  "))
print(re.sub(r"(?<=a)b", "X", "abab") if False else re.sub(r"\bthe\b", "THE", "the other the"))
