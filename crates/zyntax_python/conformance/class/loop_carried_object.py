class Pair:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Pair(self.x + other.x, self.y + other.y)


def total(n: int) -> float:
    value = Pair(1.25, -2.5)
    step = Pair(0.5, 0.25)
    for i in range(n):
        value = value + step
    return value.x + value.y


def keep_alias(n: int) -> float:
    value = Pair(1.25, -2.5)
    original = value
    step = Pair(0.5, 0.25)
    for i in range(n):
        value = value + step
    return original.x + original.y + value.x + value.y


def mutate(n: int) -> float:
    value = Pair(1.25, -2.5)
    for i in range(n):
        value.x = value.x + 0.5
        value = Pair(value.x, value.y + 0.25)
    return value.x + value.y


for n in (0, 1, 20000):
    print(total(n))
    print(keep_alias(n))
    print(mutate(n))
