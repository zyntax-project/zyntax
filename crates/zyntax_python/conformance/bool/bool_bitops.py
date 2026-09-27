# `& | ^` of two bools is a bool; with an int it is an int.
def f(a: bool, b: bool, n: int):
    print(a & b, a | b, a ^ b, str(a ^ a))
    print(a | n, a & n, a ^ n)
    print(-a, abs(b), +a)


f(True, False, 1)
f(False, True, 6)
print(True & False, True | 1, str(True ^ True), -True)
t = True
t &= False
print(t)
t |= True
print(t)
