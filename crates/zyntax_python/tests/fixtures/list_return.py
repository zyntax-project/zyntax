# A driver loop in a function that returns the list it fills: the
# interpreted frame leaves through a resume point, whose result is the
# list, and the second call runs the compiled body.
def main(n):
    times = []
    total = 0
    for i in range(n):
        total += (i * 7) & 1023
        if i % 100000 == 0:
            times.append(total)
    return times


r = main(3000000)
print(len(r), r[0], r[-1])
r2 = main(3000000)
print(len(r2), r2[-1])
