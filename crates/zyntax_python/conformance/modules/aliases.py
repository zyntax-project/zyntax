# import aliases and imports inside functions
import math as m
from math import sqrt as root, pi as PI

print(m.floor(3.7), root(81), round(PI, 3))

def area(r):
    import math
    return round(math.pi * r * r, 2)

print(area(2))

def hyp(a, b):
    from math import hypot
    return hypot(a, b)

print(hyp(6, 8))
