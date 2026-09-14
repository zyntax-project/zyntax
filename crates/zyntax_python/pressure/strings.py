# A string built, measured and dropped every step; the accumulator is
# an int so nothing keeps them.
import sys

def main(n: int) -> int:
    total = 0
    for i in range(n):
        s = "step " + str(i)
        t = s + "!"
        total += len(t)
    return total

print(main(int(sys.argv[1])))
