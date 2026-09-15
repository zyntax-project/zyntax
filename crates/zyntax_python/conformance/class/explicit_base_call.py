# Base.__init__(self, ...) and Base.method(self, ...) call the class's
# own method whatever the instance's class overrides; a dispatched call
# returns what any override may; a field first given None in a base
# holds what a subclass or a dynamic store puts in it.

class TaskState(object):
    def __init__(self):
        self.packet_pending = True
        self.task_waiting = False
        self.link = None

    def isTaskHoldingOrWaiting(self):
        return not self.packet_pending and self.task_waiting

    def fn(self, pkt):
        raise NotImplementedError

    def describe(self):
        return "state"


class Task(TaskState):
    def __init__(self, i, state):
        self.ident = i
        self.packet_pending = state.packet_pending
        self.task_waiting = state.task_waiting
        self.link = state

    def fn(self, pkt):
        if pkt is None:
            return self
        return self.link

    def describe(self):
        return "task " + str(self.ident) + " over " + TaskState.describe(self)


class Idle(Task):
    def __init__(self, i, state):
        Task.__init__(self, i, state)

    def describe(self):
        return "idle " + Task.describe(self)


s = TaskState()
s.task_waiting = True
t = Idle(3, s)
print(t.ident, t.packet_pending, t.task_waiting, t.isTaskHoldingOrWaiting())
print(t.describe())
tasks = [None] * 2
tasks[0] = t
tasks[1] = Task(4, s)
for k in tasks:
    r = k.fn(None)
    print(r is None, r.ident if r is not None else -1, k.fn(1) is s)


class Rec(object):
    def __init__(self):
        self.pending = None


class Other(object):
    def __init__(self):
        self.count = 3


def hold(r, pkt):
    if pkt is None:
        pkt = r.pending
        if pkt is None:
            return "wait"
        r.pending = None
        return "send " + str(pkt.ident)
    r.pending = pkt
    return "hold"


handles = [Rec(), Other()]
rec = handles[0]
print(hold(rec, Task(7, s)), rec.pending is None, hold(rec, None), rec.pending is None, hold(rec, None))


class Vec:
    def __init__(self, x):
        self.x = x


class Point:
    def __init__(self, x):
        self.x = x

    def pick(self):
        return self


class SubPoint(Point):
    def pick(self):
        return Vec(self.x * 10)


for thing in [Point(1.5), SubPoint(2.5)]:
    print(thing.pick().x)
