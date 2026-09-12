# test_class: attributes, methods, __init__
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def manhattan(self) -> int:
        return abs(self.x) + abs(self.y)

    def moved(self, dx: int, dy: int):
        return Point(self.x + dx, self.y + dy)

p = Point(3, -4)
print(p.x, p.y)
print(p.manhattan())
q = p.moved(1, 1)
print(q.x, q.y)
p.x = 10
print(p.x)
