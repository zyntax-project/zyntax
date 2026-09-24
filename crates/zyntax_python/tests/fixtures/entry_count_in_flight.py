class T:
    def __init__(self):
        self.k = 1


def f(n, t):
    s = 0
    i = 0
    while i < n:
        s += (i ^ t.k) % 7
        if i % 5000011 == 4:
            t.k += 1
        i += 1
    s2 = 0
    i = 0
    while i < n:
        s2 += i % 3
        i += 1
    return s * 3 + s2


t = T()
acc = 0
for r in range(3000):
    acc += f(10, t)
acc += f(60000000, t)
print(acc, t.k)
