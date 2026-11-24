# translate with a table from maketrans, and format_map.

table = str.maketrans("abc", "xyz")
print("aabbccdd".translate(table))
print("hello, world!".translate(str.maketrans("", "", "lo")))
print("hello".translate(str.maketrans("el", "ip", "o")))
print("a-b_c".translate(str.maketrans("-_", "  ")))

vowels = str.maketrans("", "", "aeiou")
print("education".translate(vowels), "rhythm".translate(vowels))
print("x".translate({ord("x"): "yy"}), "x".translate({ord("x"): None}), "x".translate({ord("x"): 66}))

data = {"name": "Ada", "year": 1815}
print("{name} was born in {year}".format_map(data))
print("{name!r:>8}|{year:05d}".format_map(data))
print("{{literal}} {name}".format_map(data))
