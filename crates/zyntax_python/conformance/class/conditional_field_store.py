# A field assigned a conditional expression whose arms read fields.

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def maximize(self, other):
        self.x = self.x if self.x > other.x else other.x
        self.y = self.y if self.y > other.y else other.y
        return self

    def pick(self, flag, other):
        self.x = other.x if flag else self.x
        return self.x


a = Point(1.0, 4.0)
b = Point(3.0, 2.0)
c = a.maximize(b)
print(c.x, c.y)
print(a.pick(True, b), a.pick(False, Point(9.0, 9.0)))
