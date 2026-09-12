# test_str: concat, len, index, slice, repeat, in
s = "hello"
print(s + " world")
print(len(s))
print(s[0], s[-1])
print(s[1:4])
print(s * 2)
print("ell" in s, "xyz" in s)
print(s == "hello", s != "hello")
print(s.upper())
print("  pad  ".strip())
print("a,b,c".split(","))
print("-".join(["x", "y", "z"]))
