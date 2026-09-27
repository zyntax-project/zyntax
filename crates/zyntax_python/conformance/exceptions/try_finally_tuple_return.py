# A tuple returned from inside try/finally, the finally still running.

class S:
    def __init__(self):
        self.depth = 0
    def colour(self, n):
        if self.depth > 3:
            return (0, 0, 0)
        try:
            self.depth = self.depth + 1
            if n < 0:
                return (0, 0, 0)
            return (n, n + 1, n + 2)
        finally:
            self.depth = self.depth - 1

def g(n):
    try:
        return (n, 2.5)
    finally:
        pass

s = S()
t = 0
for i in range(-3, 50):
    a, b, c = s.colour(i)
    t += a + b + c
print(t, s.depth)
print(g(3))
x, y = g(4)
print(x + y)
