def pick(flag):
    return "Hello World" if flag else 7
v = pick(True)
print(v.isalpha(), v.istitle(), v.swapcase(), v.casefold(), v.zfill(14), v.ljust(13, "."), v.center(15, "-"))
print(v.find("o", 5), v.rfind("o"), v.count("o", 0, 6), v.index("W"), v.startswith("World", 6), v.endswith("o", 0, 5))
print(v.replace("o", "0", 1), v.removeprefix("Hello "), v.removesuffix(" World"))
try:
    v.rindex("zz")
except ValueError as e:
    print("ValueError", e)
try:
    v.ljust(20, "ab")
except TypeError as e:
    print("TypeError", e)
