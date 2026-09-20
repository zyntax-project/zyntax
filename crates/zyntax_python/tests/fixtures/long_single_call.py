# One call, long enough that the frame resumed into the outlined region
# is still in it when the optimizing tier's resume points arrive.
def run(n):
    total = 0
    i = 0
    while i < n:
        total += (i * 7) & 1023
        i += 1
    return total


print(run(50000000))
