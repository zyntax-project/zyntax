def total(*nums):
    s = 0
    for n in nums:
        s += n
    return s
print(total(1, 2, 3))
print(total())
xs = [4, 5]
print(total(*xs))
print(total(1, *xs, 9))

def first(head, *rest):
    return head, len(rest), rest
print(first(1, 2, 3))
print(first(1))
print(first(head=7))

def tagged(tag, sep=":", *items):
    return sep.join([tag] + [str(i) for i in items])
print(tagged("a", "-", 1, 2))
print(tagged("a"))

def forward(*args):
    return total(*args)
print(forward(1, 2, 3, 4))

def kw(*args, **kwargs):
    return len(args), kwargs.get("sep", "-")
print(kw(1, 2, 3, sep="+"))
print(kw())

class A:
    def __init__(self, name, *tags):
        self.name = name
        self.tags = tags
    def f(self, *a):
        return sum(a)
    def g(self, k, *a, scale=1):
        return k * scale + sum(a)

class B(A):
    def __init__(self, *args):
        super().__init__(*args)
a = A("x", "p", "q")
print(a.name, a.tags)
b = B("y", "r")
print(b.name, b.tags)
print(a.f(1, 2, 3), a.f())
print(a.g(2, 3, 4, scale=10))
print(A.f(a, 5, 6))
