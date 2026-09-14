# A list made, grown, read and dropped every step.
import sys

def main(n: int) -> int:
    total = 0
    for i in range(n):
        xs = [i, i + 1, i + 2]
        xs.append(i + 3)
        total += xs[3] - xs[0]
    return total

print(main(int(sys.argv[1])))
