from bm_ai import n_queens, permutations

print(len(list(n_queens(6))), len(list(n_queens(7))))
print(list(n_queens(6))[0], list(permutations(range(3))))
