# Comparisons of a local that holds an int or a float by turns.
def run(flag, n, d):
    x = n
    if flag:
        x = n * 1.0
    y = d
    if not flag:
        y = d + 0.0
    print(x < y, x <= y, x > y, x >= y, x == y, x != y)
    print(x < 3, x == 7, 2.5 < x, x > 6.5, 0 < x < 100)


def wide(flag):
    x = 2**53 + 1
    if flag:
        x = 2.0**53
    y = 2.0**53
    if flag:
        y = 2**53 + 1
    print(x == y, x != y, x < y, x <= y, x > y, x >= y)
    z = 2**60
    if flag:
        z = float("nan")
    print(z == z, z < 1.0, z > 1.0, z != z)


def bools(flag):
    b = True
    if flag:
        b = 5
    print(b == 1, b < 2, b > 0.5, b == True, b >= 5)


run(True, 7, 2)
run(False, 7, 2)
run(True, 2, 7)
run(False, 7, 7)
wide(True)
wide(False)
bools(True)
bools(False)
