class Countdown:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n == 0:
            raise StopIteration
        self.n -= 1
        return self.n


c = Countdown(2)
print(next(c), next(c))
try:
    next(c)
except StopIteration:
    print("done")
g = (x for x in [1, 2])
print(next(g), next(g))
