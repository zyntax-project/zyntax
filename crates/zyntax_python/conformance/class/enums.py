from enum import Enum, IntEnum, StrEnum, auto, unique


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

    def describe(self):
        return self.name.lower() + "=" + str(self.value)

    def is_warm(self):
        return self is Color.RED


class Level(IntEnum):
    LOW = 1
    MID = 5
    HIGH = 10


class Mode(StrEnum):
    FAST = auto()
    SLOW = auto()


@unique
class Planet(Enum):
    MERCURY = auto()
    VENUS = auto()
    EARTH = auto()


print(Color.RED, repr(Color.GREEN), Color.BLUE.name, Color.BLUE.value)
print(Color(2), Color["BLUE"], Color.RED == Color.RED, Color.RED is Color.RED, Color.RED == Color.GREEN)
print(list(Color), len(Color), [c.name for c in Color], [c.value for c in Color])
for c in Color:
    print(c.describe(), c.is_warm())
print(isinstance(Color.RED, Color), Color.RED in list(Color))
print({Color.RED: "r", Color.BLUE: "b"}[Color.BLUE])
try:
    Color(9)
except ValueError as e:
    print("ValueError", e)
try:
    Color["PURPLE"]
except KeyError as e:
    print("KeyError", e)


def paint(color):
    if color == Color.RED:
        return "warm"
    return "cool"


print(paint(Color.RED), paint(Color.GREEN), paint(Color(3)))
print(Level.LOW < Level.HIGH, Level.MID == 5, Level.HIGH > 7, int(Level.MID), Level(10), sorted([Level.HIGH, Level.LOW]))
print(Level.MID, repr(Level.MID), Level.LOW + 1, 3 * Level.MID, Level.HIGH // 4, Level.MID - Level.LOW)
print(Mode.FAST, Mode.SLOW.value, repr(Mode.FAST), Mode("slow"))
print([p.value for p in Planet], Planet.EARTH.name, Planet(2))
match Color.GREEN:
    case Color.RED:
        print("is red")
    case Color.GREEN:
        print("is green")
