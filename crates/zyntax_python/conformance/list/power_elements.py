# An integer power as an element of a list, tuple, dict or set display,
# at module level and inside a function.
print([2**53 - 1, 5])
xs = [2**3, 1]
print(xs)


def g():
    print([2**40, 7])
    ys = [3**4, 2**10, 5**0]
    print(ys)
    print((2**3, 5))
    print({2**3: 1})
    print(sorted({2**3, 3**2}))
    n = 5
    print([n**2, n**3, 2**n])


g()
print([2**i for i in range(5)])
print([(-2)**3, 2**-1, 2.0**3])
