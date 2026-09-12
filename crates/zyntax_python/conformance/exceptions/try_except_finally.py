# test_exceptions: raise, except, else, finally ordering
def risky(n: int) -> int:
    if n < 0:
        raise ValueError("negative")
    return n * 2

try:
    print(risky(3))
    print(risky(-1))
    print("not reached")
except ValueError as e:
    print("caught", e)
else:
    print("no error")
finally:
    print("finally")

try:
    print(1 // 0)
except ZeroDivisionError:
    print("div by zero")
