class Planner(object):
    def __init__(self):
        self.mark = 0

    def bump(self):
        self.mark += 1
        return self.mark


planner = None
count = 0


def setup():
    global planner, count
    planner = Planner()
    count += 1


def current():
    return planner


setup()
p = planner
print(p is None, p.mark, count)
p.bump()
print(planner.mark, current().bump())
setup()
q = planner
print(q.mark, count, p.mark)
