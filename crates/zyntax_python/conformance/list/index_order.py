xs = [1]


def f():
    xs.append(2)
    return 1


def g():
    xs.append(3)
    return 2


print(xs[f()])
xs[g()] = 30
print(xs)


def local_order():
    ys = [10]

    def grow():
        ys.append(20)
        return -1

    print(ys[grow()])
    ys[grow()] = 5
    print(ys)


local_order()
