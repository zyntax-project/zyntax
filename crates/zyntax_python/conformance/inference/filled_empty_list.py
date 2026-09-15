# An empty list literal takes the kind of what the body puts in it:
# appends, inserts, extends, += and element stores of one kind make a
# typed list; a second kind, an alias, a call argument or a nested
# scope keep it a list of anything.

def strings():
    parts = []
    for i in range(5):
        parts.append("p" + str(i))
    parts.insert(0, "start")
    parts.extend(["x", "y"])
    parts += ["z"]
    parts[1] = "one"
    print(",".join(parts), len(parts), parts[0], "y" in parts, parts.pop())
    for p in parts:
        print(p.upper(), end=" ")
    print()


def numbers():
    nums = []
    i = 0
    while i < 6:
        nums.append(i * i)
        i += 1
    nums[2:4] = [9, 9]
    print(sum(nums), max(nums), sorted(nums), nums.index(9))
    if nums:
        print("full")
    if not nums:
        print("empty")


def mixed():
    xs = []
    xs.append(1)
    xs.append("two")
    print(xs)


def aliased():
    xs = []
    ys = xs
    ys.append(2.5)
    xs.append(1)
    print(xs, ys)


def passed():
    def fill(zs):
        zs.append("s")
    xs = []
    xs.append(1)
    fill(xs)
    print(xs)


def captured():
    xs = []
    xs.append(1)
    f = lambda: xs.append("s")
    f()
    print(xs)


def rebound():
    xs = []
    xs.append(1)
    xs = [2.5]
    print(xs)


def twice():
    xs = []
    xs.append("a")
    xs = []
    xs.append("b")
    print(xs)


strings()
numbers()
mixed()
aliased()
passed()
captured()
rebound()
twice()
