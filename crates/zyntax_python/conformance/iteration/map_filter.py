# map and filter over lists, ranges and strings, with lambdas, defs and builtins
xs = [1, 2, 3, 4, 5, 6]
print(list(map(lambda x: x * x, xs)))
print(list(filter(lambda x: x % 2 == 0, xs)))

def double(x):
    return x * 2

print(list(map(double, xs)))
print(list(map(str, xs)))
print(list(map(int, ["1", "2", "3"])))
print(sum(map(len, ["a", "bb", "ccc"])))
words = ["", "hi", "", "there"]
print(list(filter(None, words)))
print(list(filter(len, words)))
print(list(map(lambda a, b: a + b, [1, 2], [10, 20])))
for sq in map(lambda x: x ** 2, range(4)):
    print(sq)
print(list(map(float, xs))[0])
print(list(map(abs, [-1, -2, 3])))
print(list(filter(lambda w: len(w) > 2, words)))
print(len(list(filter(bool, [0, 1, "", "a", None, []]))))
print(sorted(map(lambda s: s.upper(), ["b", "a"])))
