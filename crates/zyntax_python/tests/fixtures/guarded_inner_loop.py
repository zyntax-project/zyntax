# Each pass of the outer loop runs an inner loop over a range that
# starts at the outer counter, skipped when that range is empty, and
# reads after it what the pass computed before it. The outer loop
# passes its header too few times to ask for a resume point, so the
# frame leaves the interpreter through the inner loop's header.
def run(n):
    total = 0
    for j in range(3):
        base = j * 7
        for i in range(j, n):
            total += i * 3 + base
        total += base
    return total


print(run(200000))
print(run(2))
