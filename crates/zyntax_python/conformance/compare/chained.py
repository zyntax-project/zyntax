# chained comparisons evaluate each operand once
x = 5
print(1 < x < 10)
print(1 < x < 3)
print(1 <= x <= 5)
print(10 > x > 1)
print(1 < x > 3)
print(x == 5 == 5)
def once(v: int) -> int:
    print("once", v)
    return v
print(1 < once(2) < 3)
