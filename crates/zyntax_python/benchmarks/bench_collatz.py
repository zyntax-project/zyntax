# Collatz step counts for every n below 300000: integer division,
# remainder and a branch per step, in a loop whose trip count depends
# on the data.
# Returns 35669673.

def main() -> int:
    total = 0
    for n in range(1, 300000):
        x = n
        steps = 0
        while x > 1:
            if x % 2 == 0:
                x = x // 2
            else:
                x = 3 * x + 1
            steps += 1
        total += steps
    return total

print(main())
