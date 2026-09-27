# An integer power as an element of a list literal is the power.
print([2**53 - 1, 5])
xs = [2**3, 1]
print(xs)


def g():
    print([2**40, 7])


g()
print([2**3])
print([[2**2, 3], [4, 3**3]])
print([1 if 2**3 > 7 else 0, 2**3])
