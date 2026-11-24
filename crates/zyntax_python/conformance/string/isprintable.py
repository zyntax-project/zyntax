print("abc".isprintable(), "a\tb".isprintable(), "".isprintable(), " ".isprintable())
print("a b".isprintable(), "é".isprintable(), "\x7f".isprintable(), "a\nb".isprintable())
print("​".isprintable(), "日本".isprintable())
vals = ["ok", "no\n", 5]
print([v.isprintable() for v in vals if isinstance(v, str)])
print("x­x".isprintable())
