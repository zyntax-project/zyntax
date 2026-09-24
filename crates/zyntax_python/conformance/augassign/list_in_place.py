# test_augassign.AugAssignTest.testInList / testInDict: `+=` and `*=` on
# a list change the list itself, so every name bound to it sees the
# change; on an int, a str and a tuple they rebind the name alone.
xs = [1]
alias = xs
xs += [2, 3]
print(xs, alias, xs is alias)
xs += (4,)
print(xs, alias, xs is alias)
xs *= 2
print(xs, alias, xs is alias)
xs *= 1
print(xs, alias)
xs *= 0
print(xs, alias, xs is alias)
xs += [7]
xs *= -3
print(xs, alias)

grid = [[0], [1]]
row = grid[0]
grid[0] += [9]
grid[1] *= 3
print(grid, row)


class Bag:
    def __init__(self):
        self.items = [1, 2]


bag = Bag()
seen = bag.items
bag.items += [3]
bag.items *= 2
print(bag.items, seen)

n = 1
m = n
n += 1
print(n, m)
s = "a"
t = s
s += "b"
s *= 2
print(s, t)
p = (1,)
q = p
p += (2,)
p *= 2
print(p, q)


def grow(items, k):
    items += [k]
    items *= 2
    return items


base = [0]
out = grow(base, 5)
print(base, out, base is out)
