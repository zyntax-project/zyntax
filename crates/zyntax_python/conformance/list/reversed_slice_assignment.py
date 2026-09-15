# Every shape of xs[a:b] = xs[c:d:-1]: coinciding ranges, ranges of
# the same length elsewhere, different lengths, empty, out of range,
# negative bounds, and a second name for the same list.

def show(label, xs):
    print(label, xs)

xs = [0, 1, 2, 3, 4, 5]
xs[:3] = xs[2::-1]
show("prefix", xs)
k = 4
xs[:k + 1] = xs[k::-1]
show("k", xs)
xs[2:5] = xs[4:1:-1]
show("middle", xs)
xs[:] = xs[::-1]
show("whole", xs)
xs[0:2] = xs[5:3:-1]
show("same length elsewhere", xs)
xs[1:2] = xs[4:1:-1]
show("longer", xs)
xs[1:4] = xs[0:0:-1]
show("shorter", xs)
xs[3:1] = xs[5:2:-1]
show("inverted target", xs)
xs[-3:] = xs[-1:-4:-1]
show("negative", xs)
xs[:100] = xs[100::-1]
show("out of range", xs)
ys = [1.5, 2.5, 3.5]
ys[:2] = ys[1::-1]
show("floats", ys)
ws = ["a", "b", "c", "d"]
ws[1:] = ws[:0:-1]
show("strings", ws)
zs = [1, "x", 2.5, None]
zs[:] = zs[::-1]
show("mixed", zs)
n = 0
xs[:n] = xs[n - 1::-1]
show("empty", xs)
xs = [0, 1, 2]
xs[1:3] = xs
show("itself", xs)
xs[0:1] = xs[1:3]
show("own slice", xs)
