class A:
    def who(self):
        return type(self) is A

    def name(self):
        return type(self).__name__


class B(A):
    pass


x = B()
print(A().who(), B().who(), A().name(), B().name())
print(type(x) is B, type(x) is A, type(x) == B, type(x) != A, type(x) is not B)
print(type(x) is type(B()), type(x) is type(A()), type(1) is int, type(1.5) is int)
print(type("s") is str, type([1]) is list, type({}) is dict, type((1,)) is tuple, type(True) is bool)
for v in [1, "a", 2.5, [1], None]:
    if type(v) is int:
        print("int", v)
    elif type(v) is str:
        print("str", v)
    elif type(v) in (float, list):
        print("float or list", v)
    else:
        print("other", v)
