from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


class Circle:
    __match_args__ = ("r",)

    def __init__(self, r):
        self.r = r


class Square:
    def __init__(self, side):
        self.side = side


def command(text):
    match text.split():
        case ["quit"]:
            return "bye"
        case ["go", direction]:
            return "going " + direction
        case ["go", direction, *rest]:
            return "going " + direction + " and " + str(len(rest)) + " more"
        case []:
            return "empty"
        case [first, *_]:
            return "unknown " + first


for line in ["quit", "go north", "go east fast now", "", "dance a b"]:
    print(command(line))


def kind(v):
    match v:
        case None:
            return "none"
        case True:
            return "true"
        case 0:
            return "zero"
        case -1:
            return "minus one"
        case int():
            return "int"
        case float():
            return "float"
        case str() as s if len(s) > 3:
            return "long str " + s
        case str(s):
            return "str " + s
        case [a, b] if a == b:
            return "pair of equals"
        case [a, b]:
            return "pair " + str(a) + str(b)
        case (1 | 2 | 3):
            return "small"
        case {"type": "circle", "r": r}:
            return "circle " + str(r)
        case {"type": t}:
            return "typed " + t
        case _:
            return "other"


for v in [None, True, 0, -1, 7, 2.5, "hello", "hi", [4, 4], [1, 2], {"type": "circle", "r": 3}, {"type": "box"}, {1: 2}]:
    print(kind(v))


def shape(s):
    match s:
        case Point(x=0, y=0):
            return "origin"
        case Point(x, 0):
            return "on x axis at " + str(x)
        case Point(0, y):
            return "on y axis at " + str(y)
        case Point(x=px, y=py):
            return "point " + str(px) + "," + str(py)
        case Circle(r):
            return "circle " + str(r)
        case Square(side=n):
            return "square " + str(n)
        case _:
            return "unknown"


for s in [Point(0, 0), Point(3, 0), Point(0, 4), Point(5, 6), Circle(2), Square(9), 3]:
    print(shape(s))

total = 0
for item in [(1, 2), (3, 4), (5,), "x"]:
    match item:
        case (a, b):
            total += a * b
        case (a,):
            total += a
        case _:
            continue
print(total)

match [1, [2, 3]]:
    case [a, [b, c]]:
        print(a + b + c)

status = 404
match status:
    case 200 | 201:
        print("ok")
    case 404:
        print("missing")
    case _:
        print("other")
