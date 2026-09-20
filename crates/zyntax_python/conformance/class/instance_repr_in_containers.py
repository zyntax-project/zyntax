class P(object):
    def __init__(self, x):
        self.x = x
    def __str__(self):
        return "str%d" % self.x
    def __repr__(self):
        return "P(%d)" % self.x
p = P(1)
print(p, repr(p), [p], (p, 2), {1: p}, str([p]))
xs = [p, 3]
print(xs)
