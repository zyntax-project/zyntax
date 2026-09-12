# test_bool.BoolTest.test_boolean and short-circuit
print(True and False)
print(True or False)
print(False or False)
print(not (True and False))
print(True and 5)
print(False or 7)
print(0 or "fallback")
print(3 and 4)
def side(x: int) -> bool:
    print("evaluated", x)
    return x > 0
print(side(1) or side(2))
print(side(-1) and side(2))
