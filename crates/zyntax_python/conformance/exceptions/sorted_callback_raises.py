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
