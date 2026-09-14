# Instances passed to methods and functions: known ones, ones that may
# be None, and results of functions that always return an instance.

class Vec:
    def __init__(self, x: float):
        self.x = x

    def __add__(self, other):
        return Vec(self.x + other.x)

    def plus(self, other) -> "Vec":
        return Vec(self.x + other.x)

def make(x: float) -> Vec:
    return Vec(x)

def maybe(flag: bool):
    if flag:
        return Vec(1.0)
    return None

def total(n: int) -> float:
    a = Vec(1.0)
    acc = Vec(0.0)
    for i in range(n):
        acc = acc + a
        acc = acc.plus(make(0.5))
    return acc.x

print(total(4))

def with_maybe(flag: bool) -> float:
    a = Vec(2.0)
    b = maybe(flag)
    return a.plus(b).x

print(with_maybe(True))
try:
    print(with_maybe(False))
except AttributeError as e:
    print("caught", e)

def reassigned() -> float:
    a = Vec(1.0)
    a = None
    b = Vec(3.0)
    return b.plus(a).x

try:
    print(reassigned())
except AttributeError as e:
    print("caught", e)

def chain(n: int) -> float:
    v = make(1.0)
    for i in range(n):
        v = v.plus(v)
    return v.x

print(chain(3))
