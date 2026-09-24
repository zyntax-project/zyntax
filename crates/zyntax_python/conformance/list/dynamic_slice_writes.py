# a slice of a list reached dynamically is a new list that takes values of
# any kind, whatever kind the list it came from holds


def keep(xs):
    out = xs[1:]
    out.append("tail")
    return out


def whole(xs):
    out = xs[:]
    out.append(None)
    return out


def backwards(xs):
    out = xs[::-1]
    out[0] = 2.5
    return out


def mid(xs):
    return xs[1:3]


class P:
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return "P(%d)" % self.n


lists = [[1, 2, 3, 4], [1.5, 2.5, 3.5], ["a", "b", "c"], [(1, "a"), (2, "b"), (3, "c")], [P(1), P(2), P(3)]]
for xs in lists:
    print(keep(xs), whole(xs), backwards(xs))
    print(xs)

others = [[1, 2, 3, 4], ["x", "y", "z"], [(1, 2), (3, 4), (5, 6)]]
ss = mid(others[0])
ss.extend(["p", 2.5])
print(ss)
ts = mid(others[1])
ts.insert(0, 7)
ts += [(8, 9)]
print(ts)
us = mid(others[2])
us[1] = "q"
print(us, len(us))
