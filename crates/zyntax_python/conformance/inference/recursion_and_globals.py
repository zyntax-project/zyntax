# recursion, mutual recursion, and module-level values from calls
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

def even(n):
    return True if n == 0 else odd(n - 1)

def odd(n):
    return False if n == 0 else even(n - 1)

def compute(k):
    return k * 3

total = compute(4)
print(fib(10), even(10), odd(7), total + 1)

def scale(xs, factor):
    out = []
    for x in xs:
        out.append(x * factor)
    return out

data = scale([1, 2, 3], 2)
print(data, scale(data, 0.5))

def first(xs):
    return xs[0]

class Box:
    def __init__(self, v):
        self.v = v

boxes = [Box(1), Box(2), Box(3)]
print(first(boxes).v, first([Box(9)]).v)

def area(p):
    return p.v * p.v

class Wide(Box):
    pass

print(area(Box(3)), area(Wide(4)))

def gen(n):
    i = 0
    while i < n:
        yield i * i
        i += 1

print(list(gen(4)))
print(sum(gen(3)))
