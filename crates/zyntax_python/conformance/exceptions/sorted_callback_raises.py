# A library call that calls back into the program raises what the
# callback raised, before anything after it runs.
def key_of(x: int) -> int:
    if x == 2:
        raise KeyError("no key for 2")
    return -x


try:
    zs = sorted([1, 2, 3], key=key_of)
    print("after sorted key", zs)
except KeyError as e:
    print("KeyError:", e)

try:
    xs = [3, 1, 2]
    xs.sort(key=key_of)
    print("after sort", xs)
except KeyError as e:
    print("KeyError:", e)

try:
    m = max([1, 2, 3], key=key_of)
    print("after max", m)
except KeyError as e:
    print("KeyError:", e)

try:
    m = min([4, 2, 5], key=key_of)
    print("after min", m)
except KeyError as e:
    print("KeyError:", e)

try:
    ms = list(map(key_of, [1, 2, 3]))
    print("after map", ms)
except KeyError as e:
    print("KeyError:", e)


def positive(x: int) -> bool:
    if x == 2:
        raise ValueError("filter at 2")
    return x > 0


try:
    fs = list(filter(positive, [1, 2, 3]))
    print("after filter", fs)
except ValueError as e:
    print("ValueError:", e)

print("fine", sorted([3, 1, 2]))


# The library stops at the first callback that raised: nothing of the
# program runs after it, and a list sorted in place keeps every element.
log: list[int] = []


class Bad:
    def __init__(self, v: int):
        self.v = v

    def __lt__(self, other) -> bool:
        log.append(self.v)
        if len(log) == 3:
            raise ValueError("cannot compare")
        return self.v < other.v


try:
    ys = sorted([Bad(5), Bad(3), Bad(1), Bad(4), Bad(2), Bad(6)])
    print("after sorted", len(ys))
except ValueError as e:
    print("ValueError:", e, len(log))

log.clear()
bs = [Bad(5), Bad(3), Bad(1), Bad(4), Bad(2), Bad(6)]
try:
    bs.sort()
    print("after sort", len(bs))
except ValueError as e:
    print("ValueError:", e, len(log))
print("kept", sorted([b.v for b in bs]))

log.clear()
try:
    ys = sorted([5, 3, 1, 4, 2, 6], key=lambda v: Bad(v))
    print("after sorted by key", len(ys))
except ValueError as e:
    print("ValueError:", e, len(log))

log.clear()
try:
    m = min([Bad(5), Bad(3), Bad(1), Bad(4), Bad(2), Bad(6)])
    print("after min", m.v)
except ValueError as e:
    print("ValueError:", e, len(log))

seen: list[int] = []


def noted(x: int) -> int:
    seen.append(x)
    if x == 2:
        raise KeyError("no key for 2")
    return -x


for which in ["sorted", "sort", "max", "map", "filter"]:
    seen.clear()
    try:
        if which == "sorted":
            print(sorted([1, 2, 3], key=noted))
        elif which == "sort":
            zs = [1, 2, 3]
            zs.sort(key=noted)
            print(zs)
        elif which == "max":
            print(max([1, 2, 3], key=noted))
        elif which == "map":
            print(list(map(noted, [1, 2, 3])))
        else:
            print(list(filter(noted, [1, 2, 3])))
    except KeyError as e:
        print(which, "KeyError:", e, seen)


def add_noted(a: int, b: int) -> int:
    seen.append(b)
    if b == 2:
        raise KeyError("no sum with 2")
    return a + b


seen.clear()
try:
    print(list(map(add_noted, [10, 20, 30], [1, 2, 3])))
except KeyError as e:
    print("map2 KeyError:", e, seen)

from functools import reduce

seen.clear()
try:
    print(reduce(add_noted, [1, 2, 3]))
except KeyError as e:
    print("reduce KeyError:", e, seen)

seen.clear()
try:
    print(reduce(add_noted, [1, 2, 3], 0))
except KeyError as e:
    print("reduce start KeyError:", e, seen)
