# Lists made by list() and by a subclass of list, filled by appends, a
# field read through a local, and elements of related classes.
class Bag(list):
    pass


class Shape(object):
    def area(self):
        return 0.0


class Square(Shape):
    def __init__(self, s):
        self.s = s

    def area(self):
        return self.s * self.s


class Circle(Shape):
    def __init__(self, r):
        self.r = r

    def area(self):
        return 3.0 * self.r * self.r


class Scene(object):
    def __init__(self):
        self.shapes = Bag()
        self.names = list()

    def add(self, shape, name):
        self.shapes.append(shape)
        self.names.append(name)

    def total(self):
        shapes = self.shapes
        t = 0.0
        for s in shapes:
            t += s.area()
        return t


scene = Scene()
scene.add(Square(2.0), "sq")
scene.add(Circle(1.0), "ci")
scene.add(Shape(), "base")
print(scene.total(), scene.names, len(scene.shapes), list(), list([1, 2]), Bag())
xs = list()
for i in range(3):
    xs.append(i * 2)
print(xs, sum(xs))
