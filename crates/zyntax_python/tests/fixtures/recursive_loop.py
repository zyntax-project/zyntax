def count(depth: int, n: int) -> int:
    total = 0
    for i in range(n):
        total += i % 7
    if depth > 0:
        total += count(depth - 1, n)
    return total


# One call, long enough that the interpreted frame asks for a resume
# point while the body has no native code yet; the resumed frame
# then calls the function.
print(count(3, 200000))
