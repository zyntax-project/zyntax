def show(*args):
    print(*args)
    print(len(args), args[0] if args else None)
    print(sum(args), max(args), min(args))
    for i, a in enumerate(args):
        print(i, a)
    print(list(args), args + (1,), 3 in args)
show(5, 2, 8)

def fmt(template, *values, sep=", "):
    return template + sep.join(str(v) for v in values)
print(fmt("<", 1, 2, 3))
print(fmt("<", 1, 2, sep="|"))

def mixed(a, b=10, *rest):
    return a, b, rest
print(mixed(1))
print(mixed(1, 2))
print(mixed(1, 2, 3, 4))

def outer():
    def inner(*xs):
        return len(xs)
    return inner(1, 2) + inner()
print(outer())

class Stack:
    def __init__(self):
        self.items = []
    def push_all(self, *vals):
        for v in vals:
            self.items.append(v)
        return self
    def size(self):
        return len(self.items)
s = Stack()
s.push_all(1, 2, 3).push_all(4)
print(s.size())
vals = [7, 8, 9]
s.push_all(*vals)
print(s.items)

def strs(*parts):
    return "".join(parts)
print(strs("a", "b", "c"), repr(strs()))

