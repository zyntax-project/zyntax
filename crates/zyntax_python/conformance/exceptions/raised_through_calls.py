# An exception raised inside a callee reaches the handler around the
# caller, whatever kind of call sits in between: a method, a
# constructor's __init__, a lambda, a nested def, a recursion, and a
# call chain in which nothing raises.

class A:
    def f(self):
        return 1

class B:
    def f(self):
        raise ValueError("b")
        return 0

def use(x):
    return x.f()

def quiet(n):
    return n + 1

def chain(n):
    return quiet(n) * 2

def deep(n):
    if n == 0:
        raise KeyError("deep")
    return deep(n - 1)

class C:
    def __init__(self, v):
        if v < 0:
            raise ValueError("neg")
        self.v = v

def mk(v):
    return C(v).v

def main():
    try:
        print(use(B()))
    except ValueError as e:
        print("caught", e)
    print(use(A()))
    print(chain(3))
    try:
        deep(3)
    except KeyError as e:
        print("caught", e)
    print(mk(2))
    try:
        mk(-1)
    except ValueError as e:
        print("caught", e)
    f = lambda x: 1 // x
    try:
        print(f(0))
    except ZeroDivisionError as e:
        print("caught", e)
    def g(x):
        return deep(x)
    try:
        g(1)
    except KeyError as e:
        print("caught", e)
    print("done")

main()
