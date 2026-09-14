# Binary trees from the Computer Language Benchmarks Game, depth 16:
# millions of short-lived trees built and walked against one that
# outlives them. The allocator row; a tree is dropped when nothing
# reaches it.
# Returns -104864.

class Node:
    def __init__(self, left, right, item: int):
        self.left = left
        self.right = right
        self.item = item

    def check(self) -> int:
        if self.left is None:
            return self.item
        return self.item + self.left.check() - self.right.check()

def build(item: int, depth: int) -> Node:
    if depth > 0:
        return Node(build(2 * item - 1, depth - 1), build(2 * item, depth - 1), item)
    return Node(None, None, item)

def main() -> int:
    min_depth = 4
    n = 16
    max_depth = n
    if min_depth + 2 > max_depth:
        max_depth = min_depth + 2
    stretch_depth = max_depth + 1
    result = build(0, stretch_depth).check()
    long_lived = build(0, max_depth)
    depth = min_depth
    while depth <= max_depth:
        iterations = 1 << (max_depth - depth + min_depth)
        check = 0
        for i in range(iterations):
            check += build(i, depth).check()
            check += build(0 - i, depth).check()
        result = result ^ check
        depth += 2
    result = result ^ long_lived.check()
    return result

print(main())
