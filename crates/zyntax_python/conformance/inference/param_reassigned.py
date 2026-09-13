# a body that assigns its parameter another type keeps it dynamic
def describe(n):
    n = "n=" + str(n)
    return n

def clamp(v, lo, hi):
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v

def widen(k):
    k = k * 1.5
    return k

print(describe(3), describe("x"))
print(clamp(5, 1, 3), clamp(-2, 1, 3), clamp(2, 1, 3))
print(widen(2), widen(3.0))
