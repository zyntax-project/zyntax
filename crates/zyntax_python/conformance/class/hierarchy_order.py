# Classes declared in an order that interleaves two hierarchies: an
# instance of any class deriving from A is accepted where an A is
# expected, whatever the declaration order, and nothing else is. A
# value that is not an A raises TypeError where CPython, which does not
# check annotations, would carry on.

class A:
    def __init__(self, name: str):
        self.name = name

    def who(self) -> str:
        return "A " + self.name

class B(A):
    def who(self) -> str:
        return "B " + self.name

class C:
    def __init__(self, name: str):
        self.name = name

class D(A):
    pass

class E(B):
    def who(self) -> str:
        return "E " + self.name

class F(C):
    pass

def take_a(x: A) -> str:
    return x.who()

def take_c(x: C) -> str:
    return x.name

xs = [A("a"), B("b"), C("c"), D("d"), E("e"), F("f")]
for x in xs:
    try:
        print(take_a(x))
    except TypeError as e:
        print("TypeError")
for x in xs:
    try:
        print(take_c(x))
    except TypeError as e:
        print("TypeError")
print(isinstance(E("e"), A), isinstance(D("d"), B), isinstance(F("f"), C), isinstance(C("c"), A))
