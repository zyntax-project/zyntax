# A field bound to [None] * n or [] in a constructor takes the kind of
# what the program stores into it, wherever that happens; a parameter
# receiving instances of two subclasses is their common base; a
# parameter assigned None keeps its class.

class Task(object):
    def __init__(self, i, area):
        self.ident = i
        self.link = area.list
        area.list = self
        area.tab[i] = self


class Idle(Task):
    def __init__(self, i, area):
        Task.__init__(self, i, area)


class Work(Task):
    def __init__(self, i, area):
        Task.__init__(self, i, area)
        self.done = 0


class Area(object):
    def __init__(self, n):
        self.tab = [None] * n
        self.list = None
        self.log = []

    def find(self, i):
        t = self.tab[i]
        if t is None:
            raise Exception("no task " + str(i))
        return t

    def note(self, t):
        self.log.append(t.ident)


def describe(t):
    if t is None:
        return "none"
    return "task " + str(t.ident)


def first_with_link(t):
    while t is not None and t.link is None:
        t = None
    return t


area = Area(4)
Idle(1, area)
Work(2, area)
Idle(3, area)
print(describe(area.find(2)), describe(area.find(1).link), area.tab[0] is None)
area.note(area.find(3))
area.note(area.list)
print(area.log, sum(area.log))
print(describe(first_with_link(area.list)), describe(first_with_link(area.find(1))))
try:
    area.find(0)
except Exception as e:
    print("Exception", e)
