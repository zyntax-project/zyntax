# test_class.ClassTests: operator methods
class Vec:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)
    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y
    def __str__(self) -> str:
        return "Vec(" + str(self.x) + ", " + str(self.y) + ")"

a = Vec(1, 2)
b = Vec(3, 4)
print(a + b)
print(a == Vec(1, 2))
print(a == b)
print(str(a))
