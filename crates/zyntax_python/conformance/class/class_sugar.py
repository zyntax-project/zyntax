class Bag(list):
    pass


class Strength(object):
    WEAK = None
    STRONG = None

    def __init__(self, level, name):
        super(Strength, self).__init__()
        self.level = level
        self.name = name

    @classmethod
    def stronger(cls, a, b):
        return a.level < b.level

    @classmethod
    def strongest(cls, a, b):
        if cls.stronger(a, b):
            return a
        return b

    @staticmethod
    def describe(level):
        return "level %d" % level

    def peer(self):
        return self.__class__.WEAK if self.level > 0 else self.__class__.STRONG


Strength.WEAK = Strength(5, "weak")
Strength.STRONG = Strength(0, "strong")


class Shape(object):
    def __init__(self, k):
        super(Shape, self).__init__()
        self.k = k

    def describe(self):
        return "%s:%d" % (self.name(), self.area())


class Square(Shape):
    def name(self):
        return "square"

    def area(self):
        return self.k * self.k


class Line(Shape):
    def name(self):
        return "line"

    def area(self):
        return 0


class Thick(Line):
    def area(self):
        return self.k


def main():
    b = Bag()
    b.append(3)
    b.append(1)
    print(b, len(b), sorted(b))
    print(Strength.stronger(Strength.STRONG, Strength.WEAK), Strength.strongest(Strength.WEAK, Strength.STRONG).name)
    print(Strength.describe(4), Strength.WEAK.peer().name, Strength.STRONG.peer().name)
    shapes = [Square(3), Line(2), Thick(4)]
    for s in shapes:
        print(s.describe())
    s = shapes[2]
    print(s.describe(), shapes[0].name())


main()
