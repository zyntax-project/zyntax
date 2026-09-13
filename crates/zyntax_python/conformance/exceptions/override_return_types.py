# An override may return another type than the method it overrides, or
# nothing but an exception. A call through the base type, or through a
# dynamic receiver, gets what the override does.

class A:
    def f(self):
        return 1

class B(A):
    def f(self):
        raise ValueError("b")

class C(A):
    def f(self):
        return "c"

def via_base(x: A):
    return x.f()

def dynamic(x):
    return x.f()

for call in [via_base, dynamic]:
    for obj in [A(), B(), C()]:
        try:
            print(call(obj))
        except ValueError as e:
            print("caught", e)
