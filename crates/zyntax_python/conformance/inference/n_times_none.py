# `n * [None]` is a list of what the program puts in it, as `[None] * n`.
class Node:
    def __init__(self, id):
        self.id = id


class Table:
    def __init__(self, count):
        self.count = count
        self.nodes = count * [None]
        for i in range(count):
            self.nodes[i] = Node(i * 10)

    def get(self, i):
        return self.nodes[i]


def main():
    t = Table(3)
    print(t.get(2).id, [n.id for n in t.nodes], len(t.nodes))
    slots = 2 * [None]
    print(slots)
    slots[0] = Node(5)
    print(slots[0].id, slots[1])


main()
