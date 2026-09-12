# test_grammar.GrammarTests.test_if_else_expr
def f(x: int) -> int:
    return x if x > 0 else -x

print(f(5))
print(f(-5))
print(1 if 1 else 0)
print(1 if 0 else 0)
print("yes" if 3 > 2 else "no")
