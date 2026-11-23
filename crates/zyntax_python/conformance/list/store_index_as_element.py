# A store whose index and element are the same value.
xs = [0] * 5
for i in range(5):
    xs[i] = i
print(xs)
ys = [0] * 4
for i in range(4):
    ys[i] = i * i
print(ys)
zs = [0.0] * 3
for i in range(3):
    zs[i] = float(i)
print(zs)
