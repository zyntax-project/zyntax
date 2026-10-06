def compare(i: int, x: object):
    print(i == x, i != x, x == i, x != i)
    try:
        print(i < x, i <= x, i > x, i >= x)
        print(x < i, x <= i, x > i, x >= i)
    except TypeError:
        print("unordered")


values = [None, False, True, -2, 3, -1.5, 3.5, float("nan"),
          float("inf"), float("-inf"), 9007199254740992.0, "three"]
for value in values:
    compare(3, value)
for value in [9007199254740992.0, 9223372036854775808.0, float("nan")]:
    compare(9007199254740993, value)
    compare(9223372036854775807, value)


class Ordered:
    def __lt__(self, other):
        print("lt")
        return True

    def __le__(self, other):
        print("le")
        return False

    def __gt__(self, other):
        print("gt")
        return False

    def __ge__(self, other):
        print("ge")
        return True


def ordering(i: int, x: object):
    print(i < x, i <= x, i > x, i >= x)
    print(x < i, x <= i, x > i, x >= i)


ordering(3, Ordered())


def number():
    print("number")
    return 3


def dynamic(x: object):
    print("dynamic")
    return x


for value in [2, 4.0, Ordered()]:
    print(number() > dynamic(value))
    print(dynamic(value) >= number())
