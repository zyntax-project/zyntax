# `type(v).__name__` for values of known and of dynamic type.
from array import array


def main():
    for op in range(2):
        try:
            if op == 1:
                raise ValueError("x")
            print(op)
        except Exception as e:
            print(type(e).__name__)


main()
print(type(1).__name__, type("s").__name__, type([1]).__name__, type((1,)).__name__)
print(type(None).__name__, type(2.5).__name__, type(True).__name__, type({1: 2}).__name__)
print(type(array("d", [1.0])).__name__)


class C:
    pass


print(type(C()).__name__)


def f(x):
    return type(x).__name__


print(f(1), f("a"), f([1]), f(C()), f(None))
try:
    raise KeyError("k")
except KeyError as e:
    print(type(e).__name__)
