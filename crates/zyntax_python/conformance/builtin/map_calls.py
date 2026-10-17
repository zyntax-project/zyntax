# map(f, xs) with a named callee: builtins, module functions, nested maps,
# unpacking, and a lazy consumer.
def square(x):
    return x * x


n, cycles = map(int, ["12", "3"])
print(n * cycles, n // cycles)
a, b = map(float, "1.5 2".split())
print(a + b)
print(list(map(square, [1, 2, 3])))
print(list(map(str, map(square, range(4)))))
print([x for x in map(abs, (-1, 2, -3))])
print(sum(map(len, ["ab", "cde", ""])))
print(sorted(map(ord, "cab")))
print(set(map(frozenset, [[1, 2], [2, 1], [3]])) == {frozenset([1, 2]), frozenset([3])})
