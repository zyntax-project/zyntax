# A small tree built, walked and dropped every step. The root is
# released; its children are held only by the root's fields, which the
# default release strategy does not follow.
import sys

class Node:
    def __init__(self, left, right, item: int):
        self.left = left
        self.right = right
        self.item = item

def build(item: int, depth: int) -> Node:
    if depth > 0:
        return Node(build(2 * item, depth - 1), build(2 * item + 1, depth - 1), item)
    return Node(None, None, item)

def check(n: Node) -> int:
    if n.left is None:
        return n.item
    return n.item + check(n.left) + check(n.right)

def main(n: int) -> int:
    total = 0
    for i in range(n):
        total += check(build(i, 4))
    return total

print(main(int(sys.argv[1])))
