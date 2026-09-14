# One instance made and dropped per step: the operator's result
# replaces the accumulator, which the release analysis frees.
import sys

class Vec:
    def __init__(self, x: float):
        self.x = x

    def __add__(self, other):
        return Vec(self.x + other.x)

def main(n: int) -> int:
    a = Vec(1.0)
    acc = Vec(0.0)
    for i in range(n):
        acc = acc + a
    return int(acc.x)

print(main(int(sys.argv[1])))
