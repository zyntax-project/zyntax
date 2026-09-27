# a list made in a function returning values of several types takes
# writes of any kind in its caller


def pick(n):
    if n > 0:
        return [1, 2, 3][1:3]
    return "s"
ss = pick(1)
b = ss
ss.extend(["p", 2.5])
print(ss, b)
ts = pick(1)
ts.insert(0, "z")
print(ts)
us = pick(1)
us[1] = "q"
print(us)
