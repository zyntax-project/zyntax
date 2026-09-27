# One object passed as two parameters: a loop reading a field through
# one and writing it through the other sees every write.
class A:
    def __init__(self, x):
        self.x = x


def f(a, b, n):
    s = 0
    for i in range(n):
        b.x = b.x + 1
        s += a.x
    return s


a = A(3)
print(f(a, a, 1000))
print(f(a, A(0), 10))
