def total(first, *n):
    return first + sum(n)
xs = [1, 2, 3]
t = (4, 5)
print(total(*xs))
print(total(*t))
class P:
    def __init__(self, a, *b):
        self.a = a
        self.b = b
    def m(self, x, *y):
        return x, y
p = P(*xs)
print(p.a, p.b)
print(p.m(*t))
print(p.m(0, *t))
