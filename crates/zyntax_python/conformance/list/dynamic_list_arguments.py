# A typed list reaching a dynamic slot and read back as a list of
# anything: through a bound method's record, through a parameter
# annotated as a list, and a box of something else raising TypeError.

def apply(f, v):
    f(v)


def take(ys: list):
    return len(ys)


xs = []
apply(xs.extend, [2])
print(xs)
ys = [1]
apply(ys.extend, [2, 3])
print(ys)
zs = ["a"]
apply(zs.extend, ["b"])
print(zs)
print(take([1, 2]), take([1.5]), take(["a", "b", "c"]), take([]))
try:
    apply(xs.extend, 5)
except TypeError:
    print("TypeError")
print(xs)
