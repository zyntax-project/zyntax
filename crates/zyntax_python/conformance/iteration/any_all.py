# any and all over lists and generator expressions
xs = [1, 2, 3]
print(any([]), all([]))
print(any([0, 0, 1]), all([1, 2, 3]), all([1, 0]))
print(any(x > 2 for x in xs), all(x > 0 for x in xs))
print(any(["", None, "x"]), all(["a", "b"]))
print(any(x % 2 == 0 for x in [1, 3, 5]))
names = ["Ada", "Bob"]
print(all(len(n) == 3 for n in names))
print(any(n.startswith("B") for n in names))
if any(x == 2 for x in xs) and not all(x == 2 for x in xs):
    print("some but not all")
print(all([True, 1, "x"]), any([False, 0, ""]))
