# A parameter typed from constructor results admits subclass instances
# wherever a call arrives from: a direct call, a boxed call through a
# dispatcher, and a hook on a boxed operand.
class Shape(object):
    def __init__(self, name):
        self.name = name

    def area(self):
        return 0

    def __eq__(self, other):
        return self.area() == other.area()

    def __lt__(self, other):
        return self.area() < other.area()


class Square(Shape):
    def __init__(self, side):
        Shape.__init__(self, "square")
        self.side = side

    def area(self):
        return self.side * self.side


class Circle(Shape):
    def __init__(self, r):
        Shape.__init__(self, "circle")
        self.r = r

    def area(self):
        return 3 * self.r * self.r


def bigger(a, b):
    if a.area() >= b.area():
        return a
    return b


def describe(s):
    return s.name + ":" + str(s.area())


class Ledger(object):
    def __init__(self):
        self.items = []

    def add(self, shape):
        self.items.append(shape)
        return len(self.items)

    def total(self):
        return sum(s.area() for s in self.items)


def main():
    a = Square(2)
    b = Circle(1)
    c = Shape("blob")
    # Direct calls pass two classes to one parameter.
    print(describe(bigger(a, b)), describe(bigger(b, a)), describe(bigger(c, a)))
    # Through a bound method value and a dynamic receiver.
    boxed = [Ledger(), 0]
    ledger = boxed[0]
    print(ledger.add(a), ledger.add(b), ledger.add(c), ledger.total())
    # Comparisons on boxed operands reach the dunders through hooks.
    shapes = [a, b, c, 1]
    print(shapes[0] == shapes[1], shapes[2] < shapes[0], shapes[1] < shapes[0])
    try:
        print(shapes[0] == shapes[3])
    except AttributeError as e:
        print("AttributeError", e)
    print(sorted([describe(s) for s in shapes[:3]]))
    print([describe(s) for s in sorted(shapes[:3])])


main()
