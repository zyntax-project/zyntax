# Tuples whose fields hold an int here and a float there.
def colour(i):
    if i == 0:
        return (0, 0, 0)
    return (i * 0.5, 1, i / 4)


def add(a, k, b):
    return (a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2])


c = colour(0)
for i in range(1, 4):
    c = add(c, 0.5, colour(i))
print(c, colour(0), colour(2), colour(0) == (0, 0, 0), colour(2) == (1.0, 1, 0.5))
seen = {colour(0): "black", colour(2): "grey"}
print(seen[(0, 0, 0)], seen[(1.0, 1, 0.5)], sorted([colour(3), colour(0), colour(1)]))
pairs = [(i, i * 0.5 if i % 2 else i) for i in range(4)]
print(pairs, max(pairs), [b for (a, b) in pairs])
