# A test settles that a field holds an instance; a store to the field,
# or a call that may store to it, unsettles it again.

class Node:
    def __init__(self, left, item: int):
        self.left = left
        self.item = item

    def clear(self):
        self.left = None

    def a(self) -> int:
        if self.left is not None:
            self.left = None
            return self.left.item
        return -1

    def b(self) -> int:
        if self.left is not None:
            self.clear()
            return self.left.item
        return -2

n = Node(Node(None, 5), 1)
try:
    print(n.a())
except AttributeError as e:
    print("caught", e)
m = Node(Node(None, 5), 1)
try:
    print(m.b())
except AttributeError as e:
    print("caught", e)
