# module variables read from methods
X = "hello"
N = 5
class C:
    def f(self):
        return X
    def g(self):
        return N + 1
def h():
    return X
print(C().f(), C().g(), h())
