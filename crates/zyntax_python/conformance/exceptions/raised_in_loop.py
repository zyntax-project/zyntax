# an error the library raises inside a loop leaves the loop at once
xs = [1, 2, 3]
total = 0
try:
    i = 0
    while i < 10:
        total += xs[i]
        i += 1
except IndexError:
    print("caught", total, i)

def f(ys: list[int]) -> int:
    t = 0
    i = 0
    while i < 10:
        t += ys[i]
        i += 1
    return t

try:
    print(f([1, 2]))
except IndexError as e:
    print("caught2", e)

d = {"a": 1}
found = 0
try:
    for k in ["a", "b", "c"]:
        found += d[k]
except KeyError as e:
    print("missing", e, found)

count = 0
try:
    for n in [4, 2, 0, 5]:
        count += 8 // n
except ZeroDivisionError:
    print("zero after", count)
print("done")
