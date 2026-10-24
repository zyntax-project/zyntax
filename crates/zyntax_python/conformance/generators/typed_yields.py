# Generators whose yields are lists or tuples of one kind hand them over
# as themselves; anything else is a dynamic value as before.
def perms(xs):
    pool = tuple(xs)
    n = len(pool)
    indices = list(range(n))
    cycles = list(range(n, 0, -1))
    yield tuple(pool[i] for i in indices)
    while n:
        for i in reversed(range(n)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices)
                break
        else:
            return


def queens(n):
    cols = range(n)
    for vec in perms(cols):
        if n == len(set(vec[i] + i for i in cols)) == len(set(vec[i] - i for i in cols)):
            yield vec


print(list(perms([1, 2, 3])))
print(len(list(queens(6))), list(queens(4)))


def rows(n):
    for i in range(n):
        yield [i * j for j in range(n)]


total = 0
for row in rows(4):
    total += sum(row) + len(row)
print(total, list(rows(2)))


def same_list():
    acc = []
    for i in range(3):
        acc.append(i)
        yield acc


seen = [r for r in same_list()]
print(seen, seen[0] is seen[2])

g = rows(3)
print(next(g), next(g), next(g), next(g, "done"))
try:
    next(g)
except StopIteration:
    print("stop")


def mixed(flag):
    yield (1, 2)
    if flag:
        yield None
    yield [3]


print(list(mixed(True)), list(mixed(False)))

for vec in perms("ab"):
    print(vec, type(vec) is tuple)
    break

words = tuple(w for w in ["x", "yy"])
print(words, sorted(perms((2, 1)))[0], max(perms([1, 2])))
