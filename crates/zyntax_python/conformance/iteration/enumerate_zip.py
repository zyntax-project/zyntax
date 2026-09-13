# enumerate and zip: pairs and triples, with and without a start
names = ["ada", "grace", "linus"]
for i, name in enumerate(names):
    print(i, name)
for i, name in enumerate(names, 1):
    print(i, name)
print(list(enumerate(["a", "b"])))
ages = [36, 45, 28]
for name, age in zip(names, ages):
    print(name, age)
print(list(zip([1, 2, 3], ["x", "y"])))
pairs = list(zip(names, ages))
print(pairs[0], len(pairs))
for a, b, c in zip([1, 2], [3, 4], [5, 6]):
    print(a + b + c)
d = dict(zip(names, ages))
print(d["grace"])
total = 0
for i, v in enumerate([10, 20, 30]):
    total += i * v
print(total)
for i, ch in enumerate("hey"):
    print(i, ch)
for x, y in zip("ab", [1, 2]):
    print(x, y)
print([i * v for i, v in enumerate([5, 6, 7])])
print(dict(enumerate(["p", "q"])))
