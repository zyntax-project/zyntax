# test_grammar.test_funcdef: defaults, recursion, early return
def power(base: int, exp: int = 2) -> int:
    result = 1
    for _ in range(exp):
        result *= base
    return result

def fact(n: int) -> int:
    if n <= 1:
        return 1
    return n * fact(n - 1)

print(power(3))
print(power(2, 10))
print(fact(10))
print(fact(20))
