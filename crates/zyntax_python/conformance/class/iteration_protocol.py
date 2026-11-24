class Bag:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class Gen:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield i * i


class Countdown:
    def __init__(self, start):
        self.cur = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur <= 0:
            raise StopIteration
        self.cur -= 1
        return self.cur + 1


class Plain:
    pass


b = Bag([3, 1, 2])
print(len(b), list(b), sum(b), sorted(b), 2 in b, max(b))
for x in b:
    print("bag", x)
print([x * 2 for x in b])
g = Gen(4)
print(list(g), sum(g))
for x in g:
    print("gen", x)
c = Countdown(3)
print(list(c))
for x in Countdown(2):
    print("count", x)
print(tuple(Bag([1, 2])), set(Bag([1, 1, 2])), list(map(str, Bag([5, 6]))))
a, bb, cc = Bag([7, 8, 9])
print(a, bb, cc)
items = [Bag([1]), Gen(2), Countdown(2), [9]]
for it in items:
    print(list(it))
print(list(zip(Bag([1, 2]), Gen(3))), dict(enumerate(Bag(["x", "y"]))))
try:
    for x in Plain():
        pass
except TypeError as e:
    print("TypeError", e)
try:
    list(Plain())
except TypeError as e:
    print("TypeError", e)
