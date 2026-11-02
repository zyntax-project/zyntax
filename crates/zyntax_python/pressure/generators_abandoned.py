# A generator started, left unfinished and dropped every step.
import sys

def three(i: int):
    yield i
    yield i + 1
    yield i + 2

def main(n: int) -> int:
    total = 0
    for i in range(n):
        it = three(i)
        total += next(it)
    return total

print(main(int(sys.argv[1])))
