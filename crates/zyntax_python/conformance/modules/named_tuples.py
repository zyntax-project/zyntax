from collections import namedtuple
from typing import NamedTuple

Point = namedtuple("Point", ["x", "y"])
Person = namedtuple("Person", "name age")
Config = namedtuple("Config", "host port debug", defaults=("localhost", 80, False))


class Pair(NamedTuple):
    left: int
    right: int = 10

    def total(self):
        return self.left + self.right


p = Point(1, 2)
q = Point(y=5, x=4)
print(p, q, p.x + q.y, repr(p), str(q))
x, y = p
print(x, y, p[0], p[1], p[-1], len(p), list(p), tuple(q), sum(p))
print(p == Point(1, 2), p != q, p == (1, 2), p < q, sorted([q, p, Point(0, 9)]))
print(p._replace(x=10), p._replace(y=0, x=-1), p._asdict(), Point._fields)
a = Person("Ann", 30)
print(a.name, a.age, a)
print(Config(), Config("example.org"), Config("h", 1, True), Config(port=8080))
pair = Pair(3)
print(pair, pair.left, pair.right, pair.total(), Pair(1, 2).total(), pair == Pair(3, 10))
people = [Person("Zed", 20), Person("Amy", 31), Person("Bob", 25)]
people.sort(key=lambda person: person.age)
print([person.name for person in people], max(people, key=lambda person: person.age))
for name, age in people:
    print(name, age)
lookup = {person.name: person for person in people}
print(sorted(lookup), lookup["Amy"].age)
try:
    p[2]
except IndexError as e:
    print("IndexError", e)
match p:
    case Point(x=0, y=_):
        print("on y axis")
    case Point(a, b):
        print("point", a, b)
