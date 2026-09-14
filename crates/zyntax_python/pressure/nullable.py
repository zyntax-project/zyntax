# An instance or None every step, tested and read: the value merges a
# null with an allocation, which is still released.
import sys

class Node:
    def __init__(self, v: int):
        self.v = v

def maybe(i: int):
    if i % 3 == 0:
        return None
    return Node(i)

def main(n: int) -> int:
    total = 0
    for i in range(n):
        x = maybe(i)
        if x is not None:
            total += x.v
    return total

print(main(int(sys.argv[1])))
