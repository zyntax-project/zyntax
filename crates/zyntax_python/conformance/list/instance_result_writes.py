# a function called with lists of different kinds returns a slice its
# caller can write any kind into


def mid(xs):
    return xs[1:3]


ss = mid([1, 2, 3, 4])
ss.extend(["p", 2.5])
print(ss)
ts = mid(["x", "y", "z"])
ts.insert(0, 7)
ts += [(8, 9)]
print(ts)
us = mid([(1, 2), (3, 4), (5, 6)])
us[1] = "q"
print(us, len(us))
