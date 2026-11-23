def pick(flag):
    return "a b" if flag else [1]
v = pick(True)
sp = v.split
print(sp(), sp(" "))
xs = [[1], [2]]
a = xs[0].append
a(5)
print(xs)
print(sorted(["b", "A", "c"], key=str.lower))
print(list(map(str.upper, ["a", "b"])))
print(list(map(str.strip, [" a ", "b "])))
print(list(map(str.split, ["a b", "c d"])))
print(sorted(["bb", "a", "ccc"], key=len))
print(list(map(dict.get, [{"k": 1}, {"j": 2}], ["k", "z"])))
print(list(map(list.copy, [[1], [2]])))
