def f_grow():
    xs = [1, 2]
    n = 0
    for x in xs:
        n += 1
        if len(xs) < 5:
            xs.append(0)
    return n


def pop_during():
    xs = [1, 2, 3, 4, 5, 6]
    seen = []
    for x in xs:
        seen.append(x)
        xs.pop()
    return seen, xs


def enumerate_grow():
    xs = [10, 20]
    out = []
    for i, x in enumerate(xs, 1):
        out.append((i, x))
        if len(xs) < 4:
            xs.append(x + 1)
    return out


def floats_continue():
    xs = [0.5, 1.5, 2.5, 3.5]
    total = 0.0
    for x in xs:
        if x > 2.0:
            continue
        total += x
    return total


def tuple_items(t):
    s = 0
    for v in t:
        s += v
    return s


def strings(words):
    out = ""
    for w in words:
        if w == "stop":
            break
        out += w
    else:
        out += "!"
    return out


def gen(xs):
    for x in xs:
        yield x * 2
        if len(xs) < 4:
            xs.append(x)


def nested():
    grid = [[1, 2], [3, 4, 5], []]
    total = 0
    for row in grid:
        for v in row:
            total += v
    return total


class Node:
    def __init__(self, v):
        self.v = v


def instances():
    ns = [Node(1), Node(2)]
    t = 0
    for n in ns:
        t += n.v
        if t < 5:
            ns.append(Node(t))
    return t


print(f_grow())
print(pop_during())
print(enumerate_grow())
print(floats_continue())
print(tuple_items((1, 2, 3)))
print(strings(["a", "b"]), strings(["a", "stop", "c"]))
print(list(gen([1, 2])))
print(nested())
print(instances())
