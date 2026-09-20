# An instance variable checked once stays known through statements that
# do not assign it, and stops being known where one does: in a loop
# body, a branch, or after the loop.
class C:
    def __init__(self, v):
        self.v = v
        self.link = None


def chain(n):
    head = None
    for i in range(n):
        c = C(i)
        c.link = head
        head = c
    return head


def walk(t):
    total = 0
    while t is not None:
        total += t.v
        t = t.link
    return total


def loop_kills(t):
    out = []
    for i in range(3):
        try:
            out.append(t.v)
        except AttributeError:
            out.append(-1)
        if i == 1:
            t = None
    return out


def loop_kills_every_pass(t):
    out = []
    for i in range(3):
        try:
            out.append(t.v)
        except AttributeError:
            out.append(-1)
        t = None
    return out


def branch_kills(t, flag):
    if flag:
        t = None
    try:
        return t.v
    except AttributeError:
        return -2


def kept_through_branches(t):
    x = t.v
    if x > 0:
        y = 1
    else:
        y = 2
    return t.v + y


print(walk(chain(5)))
print(loop_kills(C(7)))
print(loop_kills_every_pass(C(7)))
print(branch_kills(C(1), True), branch_kills(C(1), False))
print(kept_through_branches(C(3)))
