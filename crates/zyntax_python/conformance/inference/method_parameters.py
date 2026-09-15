# A method's parameters are typed by the calls in view, as a module
# function's are, unless a method of its name is called on a receiver
# whose class is not known: then every method of that name keeps
# dynamic parameters, so duck typing across classes still works.

class Point:
    def __init__(self, x):
        self.x = x

    def maximize(self, other):
        self.x = self.x if self.x > other.x else other.x
        return self

    def describe(self, other):
        return "point " + str(self.x) + " vs " + str(other.x)


class Vec:
    def __init__(self, x):
        self.x = x

    def describe(self, other):
        return "vec " + str(self.x) + " vs " + str(other.x)


class Sub(Point):
    def maximize(self, other):
        self.x = self.x + other.x
        return self


def typed():
    a = Point(1.5)
    b = Point(2.5)
    print(a.maximize(b).x)
    s = Sub(1.0)
    print(s.maximize(b).x)
    p = s
    print(p.maximize(a).x)


def ducks():
    things = [Point(1), Vec(2)]
    for t in things:
        print(t.describe(Vec(9)))
        print(t.describe(Point(8)))


typed()
ducks()
