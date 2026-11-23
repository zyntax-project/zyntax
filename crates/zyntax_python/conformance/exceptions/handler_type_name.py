# The type of a caught exception is the class that was raised, not the
# class the handler names, and `type(v).__name__` is its name.
class A(Exception):
    pass


class B(A):
    pass


try:
    raise B("b")
except A as e:
    print(type(e).__name__, isinstance(e, B), isinstance(e, A))

def g(n):
    try:
        if n:
            raise ValueError("v")
        raise KeyError("k")
    except Exception as e:
        print(type(e).__name__, isinstance(e, ValueError), isinstance(e, KeyError))

g(1)
g(0)

for i in range(2):
    try:
        raise ValueError("x")
    except Exception as e:
        print(type(e).__name__, isinstance(e, ValueError))


def main():
    for op in range(2):
        try:
            if op == 1:
                raise ValueError("x")
            print(op)
        except Exception as e:
            print(type(e).__name__)


main()
