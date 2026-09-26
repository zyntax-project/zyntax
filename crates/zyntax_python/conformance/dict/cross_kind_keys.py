# Keys equal across kinds find each other: an int key by a float or a
# bool of its value, a float key by an int, a tuple key by a tuple whose
# elements are equal; anything else is absent.
d = {1: "x", 2: "y"}
print(d[1.0], 1.0 in d, 1.5 in d, "a" in d, d[True], d.get(1.5, "none"))
f = {1.0: "a"}
print(1 in f, f[1])
t = {(1, 2): 0}
print((1, 2.0) in t, (1.0, 2) in t, (1, 2.5) in t, (1, "2") in t)

# The same probes against dicts large enough to be indexed.
big = {}
for i in range(40):
    big[i] = "v" + str(i)
print(big[3.0], 3.0 in big, 3.5 in big, "3" in big, big[True], big.get(2.5, "none"))
bigf = {}
for i in range(40):
    bigf[i + 0.5] = i
    bigf[float(i)] = -i
print(7 in bigf, bigf[7], bigf[7.5], 2 ** 60 in bigf)
bigt = {}
for i in range(20):
    bigt[(i, i + 1)] = i
print((3, 4.0) in bigt, (3.0, 4) in bigt, (3, 4.5) in bigt, (3, "4") in bigt)

# A store under an equal key of another kind keeps the first key.
k = {1: "a"}
k[1.0] = "b"
k[True] = "c"
print(k, len(k))
try:
    print(d[1.5])
except KeyError:
    print("KeyError")
try:
    print(d["1"])
except KeyError as e:
    print("KeyError", e)
