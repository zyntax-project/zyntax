# functools.reduce with an operator function, a function, a lambda and
# a start value; and without a start value.
import functools
import operator
from functools import reduce
from operator import mul


def add2(a, b):
    return a + b


xs = [1, 2, 3, 4]
fs = [0.5, 1.5, 2.0]
print(reduce(operator.add, xs, 0), reduce(mul, xs, 1), reduce(add2, xs, 10))
print(reduce(operator.add, fs, 0), reduce(operator.mul, fs, 1.0), reduce(operator.add, [], 7))
print(reduce(lambda a, b: a * 10 + b, xs, 0), reduce(operator.add, ["a", "b"], ""))
print(reduce(operator.sub, xs, 100), reduce(operator.floordiv, [2, 3], 100), reduce(operator.pow, [2, 2], 3))
print(functools.reduce(operator.or_, [1, 2, 4], 0), reduce(operator.add, (1, 2), 0), reduce(operator.add, xs))


class Totals(object):
    def __init__(self, n):
        self.parts = []
        for i in range(n):
            self.parts.append(i * 3)
        self.total = reduce(operator.add, self.parts, 0)


t = Totals(4)
print(t.parts, t.total)
