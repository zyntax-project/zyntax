class Fib:
    def __init__(self):
        self.a = 0
        self.b = 1

    def __iter__(self):
        return self

    def __next__(self):
        v = self.a
        self.a, self.b = self.b, self.a + self.b
        return v


class Naturals:
    def __iter__(self):
        n = 0
        while True:
            print("yield", n)
            yield n
            n += 1


for x in Fib():
    if x > 50:
        break
    print(x)

for n in Naturals():
    if n >= 2:
        break
    print("got", n)

total = 0
for x in Fib():
    total += x
    if total > 100:
        break
print(total)
