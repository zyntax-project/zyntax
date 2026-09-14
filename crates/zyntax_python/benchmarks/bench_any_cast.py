# A float boxed into a dynamically typed local and read back as a
# float, a million times: the cost of one box and one checked read.
# Returns 2500000.

from typing import Any

def main() -> int:
    total = 0.0
    for i in range(1000000):
        boxed: Any = 2.5
        v: float = boxed
        total = total + v
    return int(total)

print(main())
