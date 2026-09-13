# Calling a lambda held in a local. A function value is a record
# called through the shared calling shape: every argument boxed, the
# result boxed, the body's arithmetic dynamic. Ten million iterations
# rather than the hundred million of the direct-call kernels, since
# this is the shape that has not been made fast. Returns 35000000.

def main() -> int:
    step = lambda acc, i: acc + i % 8
    total = 0
    i = 0
    while i < 10000000:
        total = step(total, i)
        i += 1
    return total

print(main())
