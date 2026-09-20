# operator's arithmetic as functions and functools.reduce
import operator
from functools import reduce

xs = [3, 4, 5]
print(reduce(operator.add, xs, 0), reduce(operator.mul, xs), operator.sub(10, 3), operator.add(1.5, 2))
print(reduce(lambda a, b: a * 10 + b, [1, 2, 3]), operator.floordiv(17, 5), operator.mod(-7, 3))
print(operator.and_(12, 10), operator.or_(12, 10), operator.xor(12, 10), operator.lshift(1, 10))
print(reduce(operator.add, ["a", "b", "c"]), reduce(operator.mul, [2.0, 3.0]))


def total(ys):
    return reduce(operator.add, ys, 0)


print(total([1, 2, 3, 4]), total([]))
