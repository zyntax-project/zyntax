def outer(n):
    k = 10
    def g(a, b):
        i = 0
        while i < a:
            yield i + b + k
            i += 1
    total = 0
    for v in g(n, 100):
        total += v
    for v in g(2, 0):
        total += v
    gen = g(1, 5)
    return total, list(gen)

def counter():
    def count():
        c = 0
        while True:
            yield c
            c += 1
    it = count()
    return [next(it), next(it), next(it)]

print(outer(3))
print(counter())
