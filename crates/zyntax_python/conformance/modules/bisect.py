from bisect import bisect, bisect_left, bisect_right
from bisect import insort, insort_left, insort_right
import bisect as b

xs = [1, 2, 2, 4]
print(bisect_left(xs, 2), bisect(xs, 2), bisect_right(xs, 2))
print(b.bisect_left(xs, 2, 2), b.bisect_right(xs, 2, 0, 3))
insort_left(xs, 2)
insort(xs, 3)
insort_right(xs, 2)
print(xs)

words = ["a", "c", "c", "e"]
print(bisect_left(words, "c"), bisect(words, "c"))
b.insort_left(words, "b")
print(words)

floats = [1.0, 2.0, 2.0, 4.0]
print(bisect_left(floats, 2.0), bisect_right(floats, 2.0))

dynamic: list = [1, 2, 2, 4]
print(bisect_left(dynamic, 2), bisect_right(dynamic, 2))
insort(dynamic, 3)
print(dynamic)

def through_default(items, value, search=bisect):
    return search(items, value)

print(through_default(words, "c"))
print(bisect([], 3))
try:
    bisect(xs, 2, -1)
except ValueError:
    print("negative lo")
