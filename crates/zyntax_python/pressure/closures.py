# A lambda over a loop variable made and called every step.
import sys

def apply(f, v: int) -> int:
    return f(v)

def main(n: int) -> int:
    total = 0
    for i in range(n):
        f = lambda x: x + i
        total += apply(f, 1)
    return total

print(main(int(sys.argv[1])))
