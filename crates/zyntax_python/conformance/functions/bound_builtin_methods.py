w = "a b,c"
sp = w.split
print(sp(), sp(","), sp(",", 1))
up = w.upper
print(up())
print(list(map("-".join, [["a", "b"], ["c"]])))
print(list(map(w.count, ["a", "b", "z"])))
print(list(map(w.startswith, ["a", "b"])))
f = w.find
print(f("b"), f("b", 3), f("z"))
d = {"k": 1, "j": 2}
g = d.get
print(g("k"), g("z"), g("z", 7))
print(list(map(d.get, ["k", "j", "q"])))
it = d.items
print(sorted(it()))
pop = d.pop
print(pop("k"), pop("zz", 0), d)
s = {1, 2}
ad = s.add
ad(9)
print(sorted(s))
print(list(map(s.issubset, [{1, 2, 9, 5}, {1}])))
try:
    sp(1, 2, 3)
except TypeError as e:
    print("TypeError", e)
try:
    g()
except TypeError as e:
    print("TypeError", e)
