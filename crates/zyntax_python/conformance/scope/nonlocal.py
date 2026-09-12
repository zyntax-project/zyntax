# test_scope.ScopeTests.testNonLocalFunction
def counter():
    count = 0
    def inc() -> int:
        nonlocal count
        count += 1
        return count
    return inc

c = counter()
print(c())
print(c())
print(c())
d = counter()
print(d())
