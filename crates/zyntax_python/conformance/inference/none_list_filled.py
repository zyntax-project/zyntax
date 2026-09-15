# [None] * n filled with instances is a list of them, and stays one when
# passed to a function that only reads it or appends more of the same;
# a function that appends another kind keeps it a list of anything.

class Point:
    def __init__(self, i):
        self.v = i * 1.5


def largest(points):
    best = points[0]
    for p in points[1:]:
        if p.v > best.v:
            best = p
    return best


def grow(points):
    points.append(Point(100))


def taint(points):
    points.append("not a point")


def fill(n):
    points = [None] * n
    for i in range(n):
        points[i] = Point(i)
    return points


def build():
    points = [None] * 5
    for i in range(5):
        points[i] = Point(i)
    grow(points)
    print(largest(points).v, len(points))


def mixed():
    points = [None] * 3
    for i in range(3):
        points[i] = Point(i)
    taint(points)
    print(points[3], len(points))


def sparse():
    slots = [None] * 3
    slots[1] = Point(7)
    print(slots[0], slots[1].v, slots[2])


build()
mixed()
sparse()
print(largest(fill(4)).v)
