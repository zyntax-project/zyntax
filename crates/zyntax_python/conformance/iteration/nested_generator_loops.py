xs = ([x, x + 1] for x in [1, 2])
print([[y for y in ys] for ys in xs])

xs = ([x, x + 1] for x in [1, 2])
out = []
for ys in xs:
    inner = []
    for y in ys:
        inner.append(y)
    out.append(inner)
print(out)
