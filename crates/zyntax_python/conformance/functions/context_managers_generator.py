import contextlib
from contextlib import contextmanager

log = []


@contextmanager
def tag(name):
    log.append("<" + name + ">")
    yield name
    log.append("</" + name + ">")


@contextlib.contextmanager
def guarded(name, fail=False):
    log.append("open " + name)
    try:
        yield len(name)
    finally:
        log.append("close " + name)


with tag("b") as t:
    log.append("in " + t)

with tag("outer") as o, tag("inner") as i:
    log.append(o + i)

with guarded("file", fail=True) as size:
    log.append("size " + str(size))

try:
    with guarded("conn") as size:
        raise ValueError("boom")
except ValueError as e:
    log.append("caught " + str(e))

for line in log:
    print(line)
