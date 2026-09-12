# test_list: literal, index, negative index, len, append, iteration
xs = [1, 2, 3]
print(xs)
print(len(xs))
print(xs[0], xs[-1])
xs.append(4)
print(xs)
xs[0] = 10
print(xs)
for x in xs:
    print(x)
print([])
print([1, "a", 2.5, True])
