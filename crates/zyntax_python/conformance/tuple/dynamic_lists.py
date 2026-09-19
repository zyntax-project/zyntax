# A list of tuples reaching a dynamic value keeps its typed storage:
# reading, writing, iterating and printing go through the frontend hooks.
def show(x):
    # x is dynamic: called with lists of tuples and of ints
    print(x, len(x), x[0], x[-1], (2, 3) in x, 7 in x)
    total = 0
    for item in x:
        if isinstance(item, tuple):
            total += item[0]
        else:
            total += item
    print(total, x == x, x == [1], list(x), sorted(x)[0], x[1:])
    x.append(x[0])
    x[0] = x[-1]
    print(x, str(x), type(x) == list, bool(x))


pairs = [(2, 3), (1, 4), (5, 6)]
show(pairs)
show([7, 8, 9])
print(pairs, len(pairs))
pairs.pop()
print(pairs)
grid = [[(i, j) for j in range(2)] for i in range(2)]
print(grid, grid[1][0], grid[0][1][1])
