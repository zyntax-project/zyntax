# sys and typing: what a script imports first
import sys
from typing import List, Dict, Optional, Tuple

def total(xs: List[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t

def lookup(d: Dict[str, int], k: str) -> Optional[int]:
    if k in d:
        return d[k]
    return None

def pair(a: int, b: str) -> Tuple[int, str]:
    return (a, b)

print(total([1, 2, 3]))
print(lookup({"a": 1}, "a"), lookup({"a": 1}, "b"))
print(pair(1, "x"))
print(sys.maxsize > 2 ** 62)
print(len(sys.argv) >= 1)
print(__name__)
if __name__ == "__main__":
    print("main")
sys.exit(0)
print("not reached")
