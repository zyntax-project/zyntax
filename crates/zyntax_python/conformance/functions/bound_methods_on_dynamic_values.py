def pick(flag):
    return "a b" if flag else [1]
v = pick(True)
sp = v.split
print(sp(), sp(" "))
w = pick(False)
ap = w.append
ap(7)
print(w)
def pick2(n):
    if n == 0:
        return {"k": 1}
    if n == 1:
        return {3, 4}
    return "xyz"
d = pick2(0)
g = d.get
print(g("k"), g("q", 0))
s = pick2(1)
ad = s.add
ad(9)
print(sorted(s))
t = pick2(2)
print(list(map(t.count, ["x", "q"])))
try:
    pick2(5).nosuch
except AttributeError as e:
    print("AttributeError", e)
