from meteor_contest import board, fps, pieces, se_nh, solve

solutions = []
solve(60, 0, frozenset(range(len(board))), [-1] * len(board), range(len(pieces)), solutions)
print(len(solutions))
if solutions:
    print(solutions[0], solutions[-1])
