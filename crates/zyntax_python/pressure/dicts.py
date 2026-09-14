# A dict made, written, read and dropped every step.
import sys

def main(n: int) -> int:
    total = 0
    for i in range(n):
        d = {"a": i, "b": i + 1}
        d["c"] = i + 2
        total += d["c"] - d["a"]
    return total

print(main(int(sys.argv[1])))
