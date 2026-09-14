# A variable, field or parameter that holds an instance may hold None:
# `is None` tells them apart, reading through None raises
# AttributeError, and None prints and tests as itself.

class Node:
    def __init__(self, left, right, item: int):
        self.left = left
        self.right = right
        self.item = item

    def depth(self) -> int:
        d = 0
        if self.left is not None:
            d = self.left.depth()
        return d + 1

def maybe(flag: bool):
    if flag:
        return Node(None, None, 1)
    return None

n = Node(Node(None, None, 2), None, 1)
print(n.depth())
print(n.left is None, n.right is None, n.left is n.left)
print(n.right)
print(bool(n), bool(n.right))
x = maybe(False)
print(x is None, x)
y = maybe(True)
print(y.item)
xs = [n, n.left]
print(xs[1].item)
try:
    print(n.right.item)
except AttributeError as e:
    print("AttributeError", e)
try:
    x.depth()
except AttributeError as e:
    print("AttributeError", e)
z = n
z = None
print(z)
