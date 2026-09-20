# A string past the pool's largest class built every step and stored
# into a field, where the last one is left for the collector: each is
# a block of its own, outside the pool. The accumulator is an int.
import sys

class Holder:
    def __init__(self, s: str):
        self.s = s

def main(n: int) -> int:
    total = 0
    base = "x" * 4096
    h = Holder("")
    for i in range(n):
        h.s = base + str(i)
        total += len(h.s)
    return total

print(main(int(sys.argv[1])))
