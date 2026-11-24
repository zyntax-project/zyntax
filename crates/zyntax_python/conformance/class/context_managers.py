class CM:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print("enter", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        print("exit", self.name, exc_type is None)
        return False


class Quiet:
    def __enter__(self):
        return 7

    def __exit__(self, exc_type, exc, tb):
        print("swallow", exc)
        return True


with CM("a") as m:
    print("body", m.name)

with CM("b"):
    print("no target")

with CM("x") as x, CM("y") as y:
    print(x.name, y.name)

with Quiet() as q:
    print("q", q)
    raise ValueError("boom")
print("after quiet")

try:
    with CM("c"):
        raise KeyError("k")
except KeyError as e:
    print("caught", e)


def early(flag):
    with CM("f") as c:
        if flag:
            return "early"
        print("late")
    return "end"


print(early(True), early(False))
for i in range(3):
    with CM("loop%d" % i):
        if i == 1:
            continue
        if i == 2:
            break
        print("iter", i)
cm = CM("held")
with cm as h:
    print(h is cm)
