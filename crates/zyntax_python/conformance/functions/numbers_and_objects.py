# Values that hold a number here and another object there: arithmetic,
# comparison and truth on each, and a call whose receiver is read before
# a dynamic argument.
class Money(object):
    def __init__(self, v):
        self.v = v
    def __add__(self, o):
        return Money(self.v + (o.v if isinstance(o, Money) else o))
    def __radd__(self, o):
        return Money(o + self.v)
    def __lt__(self, o):
        return self.v < (o.v if isinstance(o, Money) else o)
    def __gt__(self, o):
        return self.v > (o.v if isinstance(o, Money) else o)
    def __repr__(self):
        return "Money(%r)" % (self.v,)
    def __bool__(self):
        return self.v != 0

def mixed(i):
    if i == 0:
        return 3
    if i == 1:
        return 2.5
    if i == 2:
        return "s"
    if i == 3:
        return [1]
    return Money(4)

class Box(object):
    def __init__(self, x):
        self.x = x

total = 0
for i in range(5):
    v = mixed(i)
    b = Box(v)
    print(i, repr(v), b.x, bool(v), v == 3, v != 2.5)
    if i < 2:
        total = total + b.x * 2
    elif i == 2:
        print(b.x + "t", b.x * 2)
    elif i == 3:
        print(b.x + [2], b.x * 2)
    else:
        print(b.x + 1, 1 + b.x, b.x < 5, b.x > 1, 2 < b.x)
print(total)
vals = [mixed(i) for i in range(5)]
print(vals, len(vals))
print(sorted([mixed(0), mixed(1)]), max(mixed(0), mixed(1)))
z = Box(Money(0))
print(bool(z.x), not z.x)
try:
    print(mixed(2) - 1)
except TypeError as e:
    print("TypeError")


class Node(object):
    def __init__(self, links):
        self.links = links


class Graph(object):
    def __init__(self, n):
        self.nodes = [Node([j for j in range(n) if j != i]) for i in range(n)]

    def node(self, i):
        return self.nodes[i]


class Walk(object):
    def __init__(self, n):
        self.graph = Graph(n)


def degree(w, last=None):
    ids = range(3) if last is None else tuple(w.graph.node(last).links)
    t = 0
    for i in ids:
        t += len(w.graph.node(i).links)
    return t


w = Walk(4)
print(degree(w), degree(w, 2))
