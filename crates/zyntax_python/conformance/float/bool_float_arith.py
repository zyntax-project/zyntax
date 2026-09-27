# A bool in float arithmetic counts as 0 or 1.
def f(b: bool, x: float):
    print(b + x, b * x, x - b, x / (b + 1), b * 2.5, True + 0.5)


f(True, 0.5)
f(False, 2.5)
t = True
print(t + 0.5, t * 2.5, 1.5 - t)
