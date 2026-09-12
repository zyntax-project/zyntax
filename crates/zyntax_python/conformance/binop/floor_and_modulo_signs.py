# test_binop / test_int: floor division rounds toward -inf, modulo takes the divisor's sign
for a in [7, -7]:
    for b in [3, -3]:
        print(a, b, a // b, a % b)
