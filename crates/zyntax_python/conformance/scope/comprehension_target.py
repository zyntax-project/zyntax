def first(pieces):
    grid = [[[] for p in range(len(pieces))] for row in range(1)]
    for _, p in enumerate(pieces):
        print(p)
    return len(grid)

print(first([[1, 2], [3, 4]]))
