# test_binop / test_int: `//`, `%` and `**` with literal operands, for
# either sign of the other operand.
def parts(i):
    return i % 8, i // 4, i % -3, i // -3, -i % 5, i ** 2, (i - 1) ** 2


for v in [-9, -8, -1, 0, 1, 7, 8, 9]:
    print(v, parts(v))

print([i % 4 for i in range(3, -3, -1)])
print([i // 4 for i in range(3, -3, -1)])
print([i % -4 for i in range(3, -3, -1)])
print(2.5 ** 2, (-1.5) ** 2, 3 ** 2, (-3) ** 2, 0 ** 2)

x = 1.5
x **= 2
print(x)
k = 17
k //= 3
k %= 2
print(k)

try:
    print(5 // 0)
except ZeroDivisionError as e:
    print("ZeroDivisionError", e)
try:
    print(5 % 0)
except ZeroDivisionError:
    print("ZeroDivisionError")
z = 0
try:
    print(5 // z)
except ZeroDivisionError as e:
    print("ZeroDivisionError", e)
