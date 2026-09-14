# A float boxed into an Any field and read back as a float, every step.
import sys
from typing import Any

class Bag:
    def __init__(self, payload: Any):
        self.payload = payload

def main(n: int) -> int:
    total = 0.0
    for i in range(n):
        bag = Bag(1.5)
        v: float = bag.payload
        total = total + v
    return int(total)

print(main(int(sys.argv[1])))
