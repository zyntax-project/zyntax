# Deleting keys keeps the survivors in insertion order; a key stored
# again after its deletion goes to the end; popitem takes the last.
def show(d):
    print(len(d), list(d.keys()), list(d.values()))

small = {"a": 1, "b": 2, "c": 3, "d": 4}
del small["b"]
show(small)
small["b"] = 5
show(small)
print(small.popitem(), small.popitem())
show(small)

big = {}
for i in range(50):
    big[i] = i * i
for i in range(0, 50, 3):
    del big[i]
show(big)
for i in range(0, 50, 9):
    big[i] = -i
show(big)
print(big.pop(49), big.pop(1000, -1), 10 in big, 11 in big)
print(big.popitem(), big.popitem(), len(big))

# Emptied one key at a time, then filled again.
many = {}
for i in range(2000):
    many["k" + str(i)] = i
for i in range(2000):
    if i % 7 != 3:
        del many["k" + str(i)]
order = list(many)
print(len(many), sum(many.values()), order[:4])
for i in range(1990, 2010):
    many["k" + str(i)] = i
order = list(many)
print(len(many), order[-3:])
while len(many) > 2:
    many.popitem()
print(many)
many.clear()
print(many, len(many))
many["z"] = 26
print(many)

# Items and copies see the survivors only.
e = {i: str(i) for i in range(12)}
for i in range(12):
    if i % 2:
        del e[i]
print(list(e.items()))
c = e.copy()
del c[0]
print(len(e), len(c), c == e, dict(c) == c)
try:
    del e[1]
except KeyError:
    print("KeyError")
try:
    del e["1"]
except KeyError as err:
    print("KeyError", err)
