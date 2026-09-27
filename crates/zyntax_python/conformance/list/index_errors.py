import array


def reads_i(xs, i):
    try:
        return xs[i]
    except IndexError as e:
        return "IndexError: " + str(e)


def stores_i(xs, i, v):
    try:
        xs[i] = v
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def bump_i(xs, i):
    try:
        xs[i] += 1
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def reads_f(xs, i):
    try:
        return xs[i]
    except IndexError as e:
        return "IndexError: " + str(e)


def stores_f(xs, i, v):
    try:
        xs[i] = v
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def bump_f(xs, i):
    try:
        xs[i] += 1
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def reads_s(xs, i):
    try:
        return xs[i]
    except IndexError as e:
        return "IndexError: " + str(e)


def stores_s(xs, i, v):
    try:
        xs[i] = v
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def bump_s(xs, i):
    try:
        xs[i] += 1
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def reads_a(xs, i):
    try:
        return xs[i]
    except IndexError as e:
        return "IndexError: " + str(e)


def stores_a(xs, i, v):
    try:
        xs[i] = v
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def bump_a(xs, i):
    try:
        xs[i] += 1
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def reads_b(xs, i):
    try:
        return xs[i]
    except IndexError as e:
        return "IndexError: " + str(e)


def stores_b(xs, i, v):
    try:
        xs[i] = v
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


def bump_b(xs, i):
    try:
        xs[i] += 1
        return xs
    except IndexError as e:
        return "IndexError: " + str(e)


a = [1, 2, 3]
print(reads_i(a, 0), reads_i(a, -1), reads_i(a, -3), reads_i(a, 3), reads_i(a, -4))
print(stores_i(a, -1, 9), stores_i(a, 3, 0), stores_i(a, -4, 0))
print(bump_i(a, 1), bump_i(a, -2), bump_i(a, 5))
f = [1.5, 2.5]
print(reads_f(f, 1), reads_f(f, 2), stores_f(f, -2, 0.5), stores_f(f, 2, 1.0))
s = ["a", "b"]
print(reads_s(s, -1), reads_s(s, 7), stores_s(s, 5, "c"))
arr = array.array("d", [1.0, 2.0])
print(reads_a(arr, 1), reads_a(arr, 2))
print(stores_a(arr, 0, 3.0).tolist(), stores_a(arr, 2, 1.0))
ib = array.array("b", [1, 2])
print(reads_b(ib, -1), reads_b(ib, -3), stores_b(ib, 4, 1))


def tuple_read(t, i):
    try:
        return t[i]
    except IndexError as e:
        return "IndexError: " + str(e)


print(tuple_read((1, 2, 3), 1), tuple_read((1, 2, 3), 3))


def loop_sum(xs, n):
    total = 0
    try:
        for i in range(n):
            total += xs[i]
    except IndexError as e:
        print("loop", e, total)
    return total


print(loop_sum([1, 2, 3], 3), loop_sum([1, 2, 3], 5))


def loop_store(xs, n):
    try:
        for i in range(n):
            xs[i] += 1
    except IndexError as e:
        print("store loop", e)
    return xs


print(loop_store([1, 2, 3], 3), loop_store([1, 2, 3], 4))


class Box:
    def __init__(self):
        self.items = [[1, 2], [3, 4]]


b = Box()
print(b.items[1][0], b.items[-1][-1])
b.items[0][1] = 7
print(b.items)
try:
    print(b.items[2][0])
except IndexError as e:
    print("nested", e)


def narrow_store(xs, i, v):
    try:
        xs[i] = v
    except IndexError as e:
        print("IndexError", e)
    except OverflowError as e:
        print("OverflowError", e)
    return xs.tolist()


print(narrow_store(array.array("b", [1, 2]), 0, 300), narrow_store(array.array("b", [1, 2]), 9, 300))
print(narrow_store(array.array("b", [1, 2]), -1, 5))
