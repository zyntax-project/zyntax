# generator expressions are lazy
g = (x * x for x in range(5))
print(next(g))
print(next(g))
print(list(g))
print(sum(x for x in range(10)))
