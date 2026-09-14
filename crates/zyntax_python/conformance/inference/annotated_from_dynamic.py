# A local annotated with a type takes it when the value is dynamic, and
# keeps the value's own type when that is already known. As with a
# parameter's annotation, a dynamic value that is not of the annotated
# type raises TypeError where CPython would carry on with it.

from typing import Any

class Bag:
    def __init__(self, payload: Any):
        self.payload = payload

def total(bags) -> float:
    acc = 0.0
    for bag in bags:
        v: float = bag.payload
        acc = acc + v
    return acc

print(total([Bag(1.5), Bag(2.5), Bag(3)]))

n: float = 3
print(n)

def first_len(xs) -> int:
    s: str = xs[0]
    return len(s)

print(first_len(["hello", "x"]))

def as_int(bag: Bag) -> int:
    k: int = bag.payload
    return k * 2

print(as_int(Bag(21)))
try:
    print(as_int(Bag("no")))
except TypeError as e:
    print("TypeError")
