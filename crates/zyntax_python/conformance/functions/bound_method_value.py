# A method bound to a local: called direct where the receiver is the
# object the binding saw, through the record everywhere else.

class Counter:
    def __init__(self, start):
        self.n = start

    def bump(self, by):
        self.n += by
        return self.n

    def show(self):
        return "n=" + str(self.n)


def lists():
    xs = [3, 1, 2]
    ins = xs.insert
    pop = xs.pop
    app = xs.append
    ins(0, pop())
    app(9)
    print(xs, pop(0), xs)
    srt = xs.sort
    srt()
    print(xs)
    cnt = xs.count
    print(cnt(9), cnt(7))


def instances():
    c = Counter(10)
    bump = c.bump
    print(bump(1), bump(2), c.n)
    show = c.show
    print(show())
    # The record travels: passed on, it is still the method.
    def twice(f, v):
        return f(v) + f(v)
    print(twice(bump, 5), c.n)


def rebound_receiver():
    xs = [1]
    app = xs.append
    xs = [2]
    app(3)
    print(xs)


def rebound_name():
    xs = [1]
    f = xs.append
    f(2)
    f = xs.pop
    print(f(), xs)


def in_loop():
    counters = [Counter(0), Counter(10)]
    for i in range(2):
        c = counters[i]
        bump = c.bump
        bump(i + 1)
    print(counters[0].n, counters[1].n)


def param(xs):
    app = xs.append
    app(4)
    return xs


def cond(flag):
    xs = []
    if flag:
        f = xs.append
    else:
        f = xs.pop
    f(1) if flag else f()
    print(xs)


lists()
instances()
rebound_receiver()
rebound_name()
in_loop()
print(param([1, 2]))
cond(True)
xs = [5]
xs_pop = xs.pop
print(xs_pop(), xs)
