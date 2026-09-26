# A local assigned an int and a float holds whichever it was given last.


def halves(n):
    x = 0
    for i in range(n):
        x += 0.5
    print(x, isinstance(x, int), isinstance(x, float))
    return x


def mixed(flag):
    y = 1
    if flag:
        y = 2.25
    z = y * 2 + y - 1
    w = y + True
    v = y - False
    print(y, z, w, v, y * y)
    return z


def counter(n):
    total = 0
    for i in range(n):
        if i % 3 == 0:
            total = total + 1
        else:
            total = total * 1.5 - i
    return total


halves(0)
halves(3)
mixed(0)
mixed(1)
print(counter(0), counter(1), counter(7))
acc = 0.0
for r in range(200):
    acc += counter(r % 11)
print(acc)
