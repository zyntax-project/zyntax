# a function used as a value is called from anywhere; its parameters stay dynamic
def double(x):
    return x * 2

def apply(f, v):
    return f(v)

print(double(2), double("ab"))
print(apply(double, 3), apply(double, "z"))
handlers = [double]
print(handlers[0](1.5))

def sq(x):
    return x * x

print([sq(i) for i in range(4)], sq(2.5))
k = lambda v: sq(v)
print(k(3), k(1.5))

def outer():
    s = "a"
    def inner():
        return double(s)
    return inner()

print(outer())
print(sorted([3, 1, 2], key=sq))
