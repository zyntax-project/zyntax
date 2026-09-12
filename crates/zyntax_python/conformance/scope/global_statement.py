# test_scope.ScopeTests.testGlobal
total = 0

def bump(n: int):
    global total
    total += n

bump(3)
bump(4)
print(total)
