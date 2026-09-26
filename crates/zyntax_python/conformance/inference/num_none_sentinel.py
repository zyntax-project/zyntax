# None as the starting value of a number, tested with `is None`.


def last_of(xs):
    last = None
    for v in xs:
        if last is not None:
            print("prev", last)
        last = v
    if last is None:
        print("empty")
    return last


def first_big(xs, limit):
    found = None
    for v in xs:
        if found is None and v > limit:
            found = v
    return found


print(last_of([]))
print(last_of([1, 2.5, 3]))
print(first_big([1, 5, 7.5], 4), first_big([1, 2], 4))
k = None
for i in range(3):
    if k is None or i > 1:
        k = i * 0.5
print(k)
n = None
for i in range(2):
    n = i
print(n)
n = None
print(n, n is None)
