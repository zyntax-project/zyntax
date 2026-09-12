# a local shadows a global without changing it
x = 10

def f() -> int:
    x = 20
    return x

print(f())
print(x)
