# The inner loop warms inside one call: the frame leaves the interpreter
# through a resume point outlined from the body as lowered, with the
# outer loop's counter and the pair in flight carried in from the frame.
def run(n, pairs, bodies):
    total = 0.0
    for i in range(n):
        for (a, b) in pairs:
            total += a * b
        for v in bodies:
            total -= v
    return total


pairs = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
bodies = [0.5, 0.25]
print(run(30000, pairs, bodies))
print(run(3, pairs, bodies))
