# What str() of a KeyError shows: the repr of the missing key.
def show(f):
    try:
        f()
    except KeyError as e:
        print(type(e).__name__, e)


d = {(8, 8): 1, 3: "a", "s": 2.5}
show(lambda: d[(1, 2)])
show(lambda: d[7])
show(lambda: d["nope"])
show(lambda: d.pop(99))
show(lambda: d.pop("zz"))


try:
    del d[(5, 5)]
except KeyError as e:
    print("del", e)
ints = {1: 2}
show(lambda: ints[9])
strs = {"a": 1}
show(lambda: strs["b"])
show(lambda: {}.popitem())
s = {1, 2}
show(lambda: s.remove(5))
show(lambda: set().pop())


def dynamic(key):
    m = {1: "x", "k": "y"}
    return m[key]


show(lambda: dynamic(2))
show(lambda: dynamic("q"))
show(lambda: dynamic(2.5))
try:
    raise KeyError("msg")
except KeyError as e:
    print(e)
try:
    raise KeyError(5)
except KeyError as e:
    print(e)
