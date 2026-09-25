# A tuple held in a field, read through a receiver whose class is not
# known: each read yields the tuple that instance holds.
class Node(object):
    def __init__(self, pos, id, links):
        self.pos = pos
        self.id = id
        self.links = links


class Tag(object):
    def __init__(self, pair):
        self.pair = pair


nodes = []
for y in range(2):
    for x in range(3):
        nodes.append(Node((x, y), len(nodes), []))
print([n.pos for n in nodes])

by_id = len(nodes) * [None]
for n in nodes:
    by_id[n.id] = n
for n in by_id:
    (x, y) = n.pos
    n.links.append(x * 10 + y)
print([n.links for n in by_id])


def positions(items):
    out = []
    for item in items:
        a, b = item.pos
        out.append(a + b)
    return out


print(positions(by_id))

tags = [None, None]
tags[0] = Tag(("a", 1))
tags[1] = Tag(("b", 2))
for t in tags:
    name, count = t.pair
    print(name, count, t.pair)
