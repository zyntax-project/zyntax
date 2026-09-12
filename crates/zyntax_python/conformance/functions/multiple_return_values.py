# tuple return and unpacking
def divmod_(a: int, b: int):
    return a // b, a % b

q, r = divmod_(17, 5)
print(q, r)
print(divmod_(7, 2))
