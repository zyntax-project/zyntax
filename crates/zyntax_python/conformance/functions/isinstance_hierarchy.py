# isinstance against a class hierarchy: a base-typed parameter tested
# for a subclass, a base-class method testing a subclass, and sibling
# classes, each also passed as None.
class Node(object):
    def __init__(self, value):
        self.value = value

    def kind(self):
        if isinstance(self, Leaf):
            return "leaf"
        if isinstance(self, Branch):
            return "branch"
        return "node"


class Leaf(Node):
    def __init__(self, value):
        Node.__init__(self, value)
        self.weight = value * 2


class Branch(Node):
    def __init__(self, value, kids):
        Node.__init__(self, value)
        self.kids = kids


def classify(n):
    if isinstance(n, Leaf):
        return "leaf " + str(n.value)
    elif isinstance(n, Branch):
        return "branch of " + str(len(n.kids))
    elif isinstance(n, Node):
        return "plain " + str(n.value)
    return "none"


def weigh(leaf):
    if not isinstance(leaf, Leaf):
        return -1
    return leaf.weight


def only_leaves(n):
    if isinstance(n, Branch):
        return 0
    if isinstance(n, Leaf):
        return n.weight
    return 0


def main():
    nodes = [Leaf(1), Branch(2, [Leaf(3), Leaf(4)]), Node(5)]
    for n in nodes:
        print(n.kind(), classify(n))
    print(classify(None), classify(nodes[1].kids[0]))
    print(weigh(nodes[0]), weigh(nodes[1]), weigh(None))
    print(only_leaves(nodes[0]), only_leaves(nodes[1]), only_leaves(nodes[2]))
    mixed = [nodes[0], nodes[1], 3, "x", None]
    print([classify(m) if isinstance(m, Node) else "other" for m in mixed])
    print(isinstance(nodes[0], Node), isinstance(nodes[2], Leaf), isinstance(3, Leaf))


main()
