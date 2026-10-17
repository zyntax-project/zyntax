# Arithmetic on a number that may be None: the numbers computed on, None
# raising the TypeError Python does.
def maybe(i):
    if i > 1:
        return i * 0.5
    return None
def scale(v, k):
    return v * k + 1
for i in range(4):
    m = maybe(i)
    try:
        print(i, scale(2.0, m), m * 3, 1 - m, m / 2)
    except TypeError as e:
        print(i, "TypeError", e)
    try:
        print(m + None)
    except TypeError as e:
        print("TypeError", e)
