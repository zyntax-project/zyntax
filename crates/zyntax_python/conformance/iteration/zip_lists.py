# zip() over lists in a for loop: two and three sequences, uneven lengths, a list
# that grows during the loop, an empty one.
xs = [1, 2, 3, 4]
ys = [0.5, 1.5, 2.5]
for a, b in zip(xs, ys):
    print(a, b, a * b)
names = ["a", "b"]
for n, x, y in zip(names, xs, ys):
    print(n, x + y)
rows = [[1.0, 2.0], [3.0, 4.0]]
cols = [[0.0, 0.0], [0.0, 0.0]]
for src, dst in zip(rows, cols):
    dst[:] = src
print(cols)
grow = [1, 2]
seen = []
for a, b in zip(grow, [10, 20, 30, 40]):
    seen.append((a, b))
    if len(grow) < 4:
        grow.append(a + 100)
print(seen)
for a, b in zip([], [1]):
    print("never")
print(list(zip(xs, ys)))
