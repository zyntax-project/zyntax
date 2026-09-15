xs = [0, 1, 2, 3]
xs[:3] = xs[2::-1]
print(xs)

alias = xs
xs[1:2] = [8, 9, 10]
print(xs, alias)
xs[1:4] = []
print(xs, alias)
xs[2:1] = [7, 8]
print(xs)

xs = [0, 1, 2, 3, 4, 5]
xs[1:6:2] = [9, 8, 7]
print(xs)
xs[5:0:-2] = [4, 3, 2]
print(xs)
xs[::-1] = xs
print(xs)

before = xs.copy()
try:
    xs[::2] = [1]
except ValueError:
    print(xs == before)
try:
    xs[::0] = [1]
except ValueError:
    print(xs == before)

xs[1:3] = (11, 12)
print(xs)

target = [0, 1, 2]
def replacement():
    global target
    target = [7, 8, 9]
    return [4]

target[1:2] = replacement()
print(target)

holder = {"list": [1, 2, 3]}
one: list[int] = holder["list"]
one[1:] = [8, 9]
print(holder, one)
