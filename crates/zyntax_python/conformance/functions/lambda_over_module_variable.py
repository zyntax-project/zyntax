# A module variable read by a module function and by a lambda is one
# variable: both see a rebinding, whichever is passed to a parameter.
def show(f):
    try:
        f()
    except KeyError as e:
        print("caught", e)
    print("shown")


def run(f):
    return f()


d = {1: 2}
xs = [1, 2, 3]


def getter():
    return d[1]


def first():
    return xs[0]


def missing():
    return d[5]


show(lambda: d[7])
show(missing)
print(run(getter), run(lambda: d[1]))
print(run(first), run(lambda: xs[0]))


def rebind():
    global xs
    xs = [9, 8]


rebind()
print(run(first), run(lambda: xs[0]), xs)
n = 5


def bump():
    global n
    n += 1


bump()
print(run(lambda: n), n)
print(run(lambda: d.get(7)), run(getter))
