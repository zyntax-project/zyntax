def busy(n):
    x = 0
    for i in range(n):
        x += (i ^ 5) % 3
    return x


def f(n):
    s = 0
    for i in range(3000):
        s += i % 7
    s += busy(n * 4)
    k = 0
    s2 = 0
    while k < n:
        s2 += (k ^ s) % 7
        k += 1
    return s * 7 + s2


print(f(50000000))
