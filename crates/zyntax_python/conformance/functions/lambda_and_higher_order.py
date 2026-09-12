# test_grammar.test_lambdef
square = lambda x: x * x
print(square(7))
def apply(f, v: int) -> int:
    return f(v)
print(apply(lambda x: x + 1, 41))
print(apply(square, 12))
