# test_scope.ScopeTests.testSimpleNesting / testNestingPlusFreeRefToGlobal
def make_adder(n: int):
    def add(x: int) -> int:
        return x + n
    return add

add5 = make_adder(5)
print(add5(10))
add7 = make_adder(7)
print(add7(10))
print(add5(1))
