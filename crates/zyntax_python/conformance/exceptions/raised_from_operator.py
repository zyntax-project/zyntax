# An exception raised inside an operator method leaves the loop that
# applied the operator at once, whichever operator it was.

class Vec:
    def __init__(self, x: float):
        self.x = x

    def __add__(self, other):
        return Vec(self.x + other.x)

    def __lt__(self, other):
        return self.x < other.x

    def __len__(self):
        if self.x < 0:
            raise ValueError("negative length")
        return int(self.x)

def add_all(n: int) -> float:
    a = Vec(1.0)
    acc = Vec(0.0)
    for i in range(n):
        print("add", i)
        acc = acc + a
        if i == 1:
            a = None
    return acc.x

try:
    print(add_all(5))
except AttributeError as e:
    print("caught", e)

def count_below(n: int) -> int:
    limit = Vec(2.5)
    below = 0
    for i in range(n):
        print("compare", i)
        v = Vec(float(i))
        if i == 3:
            limit = None
        if v < limit:
            below += 1
    return below

try:
    print(count_below(5))
except AttributeError as e:
    print("caught", e)

def lengths(xs) -> int:
    total = 0
    for x in xs:
        total += len(x)
    return total

try:
    print(lengths([Vec(2.0), Vec(3.0)]))
    print(lengths([Vec(2.0), Vec(-1.0), Vec(3.0)]))
except ValueError as e:
    print("caught", e)

def truths(xs) -> int:
    n = 0
    for x in xs:
        if x:
            n += 1
    return n

try:
    print(truths([Vec(1.0), Vec(0.0), Vec(2.0)]))
    print(truths([Vec(1.0), Vec(-2.0)]))
except ValueError as e:
    print("caught", e)
print("done")
