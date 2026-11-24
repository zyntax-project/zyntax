head = (1, 2)
tail = [8, 9]
print([*head, 3], [0, *tail, *head])
print((*head, 3), (*tail,))
print(sorted({*head, 3, *tail}))
print(len([*"abc"]), [*range(3)])

a, *rest = [1, 2, 3]
print(a, rest)
first, *middle, last = range(5)
print(first, middle, last)
*init, end = "abc"
print(init, end)
x, *none = (7,)
print(x, none)
p, *q = "z"
print(p, q)

for k, *v in [(1, 2, 3), (4,), (5, 6)]:
    print(k, v)

def split(items):
    h, *t = items
    return h, t
print(split([4, 5, 6]), split("xy"))

try:
    u, v, *w = [1]
except ValueError as e:
    print(e)

rows = [[1, 2, 3], [4, 5, 6]]
print(list(zip(*rows)))
print([sum(c) for c in zip(*rows)])
for c0, c1 in zip(*rows[:2]):
    print(c0, c1)
nums = (3, 9, 4)
print(max(*nums), min(*nums), max(*rows[0]))
