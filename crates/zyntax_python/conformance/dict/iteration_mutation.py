# A dict or set that changes size while a loop walks it raises
# RuntimeError at the next step, whichever name, attribute or callee
# changed it; a store to an existing key or a change undone within the
# step does not.
def alias():
    d = {1: 0}
    e = d
    out = []
    try:
        for k in d:
            out.append(k)
            if k < 4:
                e[k + 1] = 0
    except RuntimeError as err:
        print("RuntimeError", err, out)


class Holder:
    def __init__(self):
        self.m = {1: 0}

    def grow(self, k):
        self.m[k + 1] = 0

    def walk(self):
        seen = []
        try:
            for k in self.m:
                seen.append(k)
                self.grow(k)
        except RuntimeError as err:
            print("RuntimeError", err, seen)


def shrink(d, k):
    del d[k]


def callee():
    d = {i: i * i for i in range(5)}
    try:
        for k, v in d.items():
            if k == 2:
                shrink(d, 4)
    except RuntimeError as err:
        print("RuntimeError", err)


def last_step():
    d = {'a': 1, 'b': 2}
    try:
        for k in d:
            if k == 'b':
                d['c'] = 3
        print("no error")
    except RuntimeError as err:
        print("RuntimeError", err, sorted(d))


def overwrite():
    d = {'a': 1, 'b': 2}
    for k in d:
        d[k] = d[k] * 10
    total = 0
    for v in d.values():
        total += v
    print(d, total)


def sets():
    s = {1, 2, 3}
    try:
        for x in s:
            s.add(x + 10)
    except RuntimeError as err:
        print("RuntimeError", err)
    t = {1, 2, 3}
    seen = 0
    for x in t:
        t.discard(99)
        seen += x
    print(seen)


def broken():
    d = {i: i for i in range(20)}
    found = -1
    for k in d:
        if k == 7:
            found = k
            break
    d[100] = 1
    for k in d.keys():
        if k % 2:
            continue
        found += 1
    print(found)


alias()
Holder().walk()
callee()
last_step()
overwrite()
sets()
broken()
