# An exception raised and caught every other step: the instance, its
# message and the pending slot all come and go.
import sys

def get(i: int) -> int:
    if i % 2 == 0:
        return i
    raise ValueError("odd")

def main(n: int) -> int:
    total = 0
    for i in range(n):
        try:
            total += get(i)
        except ValueError:
            total += 1
    return total

print(main(int(sys.argv[1])))
