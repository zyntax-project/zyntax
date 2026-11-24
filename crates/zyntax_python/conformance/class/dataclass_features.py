from dataclasses import dataclass, field
from typing import List


@dataclass
class Point:
    x: int
    y: int = 2


@dataclass(order=True)
class Version:
    major: int
    minor: int = 0


@dataclass(frozen=True)
class Key:
    a: str
    b: int


@dataclass
class Bag:
    name: str
    items: List[int] = field(default_factory=list)
    count: int = 0

    def add(self, x):
        self.items.append(x)
        self.count += 1


@dataclass
class Point3(Point):
    z: int = 3


@dataclass
class Checked:
    value: int

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("negative")
        self.double = self.value * 2


p = Point(1)
print(p, p.x, p.y, Point(1, 2) == p, Point(3) == p, p != Point(1))
print(Point(5, 6))
print(Version(1, 2) < Version(1, 3), Version(2) > Version(1, 9), Version(1) <= Version(1), sorted([Version(2), Version(1, 5), Version(1)]))
k = Key("a", 1)
print(k, k == Key("a", 1), {k: 1}[Key("a", 1)], hash(k) == hash(Key("a", 1)))
b = Bag("b")
b.add(3)
b.add(4)
print(b, Bag("c") == Bag("c"), len(b.items))
b2 = Bag("d")
print(b2.items, b.items is not b2.items)
print(Point3(1), Point3(1, 2, 9), Point3(1) == Point3(1, 2, 3))
c = Checked(4)
print(c, c.double)
try:
    Checked(-1)
except ValueError as e:
    print("ValueError", e)
print([Point(i) for i in range(2)])
