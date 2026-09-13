# constructor parameters are typed by every construction, inherited ones included
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def norm2(self):
        return self.x * self.x + self.y * self.y

class Labeled(Point):
    def __init__(self, x, label):
        super().__init__(x, 0)
        self.label = label

class Plain(Point):
    pass

p = Point(3, 4)
q = Labeled(1.5, "q")
r = Plain(2, 5)
print(p.norm2(), q.norm2(), r.norm2())
print(p.x, q.x, r.y, q.label)

class Counter:
    def __init__(self, start=10):
        self.n = start
    def bump(self):
        self.n += 1
        return self.n

c = Counter()
d = Counter(1)
print(c.bump(), d.bump())

class Tagged:
    def __init__(self, tag, weight=1.0):
        self.tag = tag
        self.weight = weight

items = [Tagged("a"), Tagged("b", 2.5), Tagged(weight=0.5, tag="c")]
for t in items:
    print(t.tag, t.weight)

class MyError(Exception):
    def __init__(self, code=7):
        super().__init__("code " + str(code))
        self.code = code

try:
    raise MyError
except MyError as e:
    print(e, e.code)
try:
    raise MyError(3)
except MyError as e:
    print(e, e.code)
