# class attributes: constants folded, variables shared, introduced by assignment
class Random(object):
    MDIG = 32
    ONE = 1
    m1 = (ONE << (MDIG - 2)) + ((ONE << (MDIG - 2)) - ONE)
    m2 = ONE << MDIG // 2
    dm1 = 1.0 / float(m1)
    LABEL = "r" + "nd"
    HALF = 7 // 2
    NEG = -7 % 3
    FLAG = MDIG > 16 and not None

    def __init__(self, seed):
        self.seed = seed

    def scaled(self, k):
        return self.dm1 * float(k) + self.m2 - self.MDIG


class Counter(object):
    count = 0
    last = None

    def __init__(self, v):
        Counter.count += 1
        Counter.last = self
        self.v = v

    def total(self):
        return self.count + self.v


class Sub(Counter):
    def __init__(self, v):
        Counter.__init__(self, v)


print(Random.m1, Random.m2, Random.dm1, Random.LABEL, Random.HALF, Random.NEG, Random.FLAG)
r = Random(3)
print(r.scaled(1000), r.LABEL)
Counter(1)
c = Counter(2)
print(Counter.count, Counter.last.v, c.count, c.total())
s = Sub(5)
print(Sub.count, s.count, Counter.last.v)
Counter.tag = 7
print(Counter.tag + c.tag)
Counter.count = 100
print(c.count)
