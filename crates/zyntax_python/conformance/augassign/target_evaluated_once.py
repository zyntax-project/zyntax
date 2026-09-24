# test_augassign.AugAssignTest.testSequences / testCustomMethods: the
# target's container and index are evaluated once, then read, then set.
calls = []


def container():
    calls.append("container")
    return xs


def index():
    calls.append("index")
    return 1


xs = [1, 2, 3]
container()[index()] += 5
print(xs, calls)


class Counted:
    def __init__(self):
        self.d = {}
        self.n = 0

    def __getitem__(self, k):
        calls.append("get " + k)
        return self.d.get(k, 0)

    def __setitem__(self, k, v):
        calls.append("set " + k)
        self.d[k] = v


c = Counted()
c["a"] += 2
c["a"] *= 3
c["b"] -= 1
print(c.d, calls)


def key():
    calls.append("key")
    return "k"


d = {"k": 1}
d[key()] -= 4
print(d, calls)


def holder():
    calls.append("holder")
    return c


holder().n += 1
holder().n *= 5
print(c.n, calls)

grid = [[1, 2], [3, 4]]


def row():
    calls.append("row")
    return 1


grid[row()][row() - 1] -= 10
grid[0][1] += 5
print(grid, calls)

words = ["a", "b"]
words[index()] += "c"
print(words, calls)
