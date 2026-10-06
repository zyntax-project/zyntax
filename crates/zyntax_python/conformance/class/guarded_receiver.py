class Counter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def step(self, i):
        return self.width + i


class Twice(Counter):
    def step(self, i):
        return self.width + 2 * i


class Inherited(Twice):
    pass


class Other:
    def __init__(self):
        self.width = 20
        self.height = 3

    def step(self, i):
        return self.width - i


class WrongArity(Counter):
    def step(self, i, extra):
        return self.width + i + extra


def run(counter, times=2):
    total = 0
    for _ in range(times):
        for i in range(counter.height):
            total += counter.step(i) + counter.width
    return total


for value in [Counter(4, 3), Twice(4, 3), Inherited(4, 3), Other(), None, "other"]:
    try:
        print(run(value), run(value, times=1))
    except AttributeError:
        print("AttributeError")

try:
    print(run(WrongArity(4, 3)))
except TypeError:
    print("TypeError")


class NoItems:
    def __init__(self):
        self.width = 2
        self.height = 2


def cannot_specialize(value):
    for _ in range(value.height):
        if value.width:
            return value[0]


try:
    print(cannot_specialize([NoItems(), None][0]))
except TypeError:
    print("TypeError")
