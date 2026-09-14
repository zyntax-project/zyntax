# A dynamically typed field: a float stored into a slot annotated Any
# and read back as a float, a million times. The boxing row.
# Returns 1500000.

from typing import Any

class Bag:
    def __init__(self, payload: Any):
        self.payload = payload

def main() -> int:
    total = 0.0
    for i in range(1000000):
        bag = Bag(1.5)
        v: float = bag.payload
        total = total + v
    return int(total)

print(main())
