# An element assignment whose value is a conditional, then more statements
# in the same function.
def put(xs, k):
    xs[0] = k if k < 63 else -1
    xs[1] = (1 << k) if k < 8 else 0
    tail = xs[0] + xs[1]
    return tail


print(put([0, 0], 3), put([0, 0], 70), put([0, 0], 62))
