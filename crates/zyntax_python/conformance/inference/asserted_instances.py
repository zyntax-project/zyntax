# A local bound and then asserted to be an instance of a class on the
# next line is that class; the assert still fails when it should.

class Rec(object):
    def __init__(self):
        self.pending = None


class Other(object):
    def __init__(self):
        self.count = 3


class DevRec(Rec):
    def __init__(self):
        Rec.__init__(self)
        self.device = 1


def handle(r, pkt):
    d = r
    assert isinstance(d, Rec)
    if pkt is None:
        pkt = d.pending
        if pkt is None:
            return "wait"
        d.pending = None
        return "send " + str(pkt)
    d.pending = pkt
    return "hold"


handles = [Rec(), Other(), DevRec()]
print(handle(handles[0], 5), handle(handles[0], None), handle(handles[0], None))
print(handle(handles[2], 7), handle(handles[2], None))
try:
    print(handle(handles[1], None))
except AssertionError:
    print("AssertionError")
