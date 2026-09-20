# random: the Mersenne Twister as CPython runs it, seeded the same way
import random

random.seed(1234)
print(random.random(), random.random())
print([random.randrange(10) for i in range(8)])
print(random.randrange(5, 50), random.randrange(0, 100, 7), random.randint(1, 6))
print(random.getrandbits(5), random.getrandbits(40), random.randrange(2**62))
random.seed(2**40 + 17)
print(random.random())
random.seed(0)
print(random.random(), random.uniform(1.0, 2.0))
random.seed(-5)
print(random.randrange(1000), random.randrange(-20, -10, 3))
random.seed(7)
print(random.choice([10, 20, 30, 40]), random.choice("abc"))


def draws(n):
    total = 0
    for i in range(n):
        total += random.randrange(1000)
    return total


random.seed(99)
print(draws(2000))
