# a custom exception class propagates through frames
class TooBig(Exception):
    pass

def check(n: int) -> int:
    if n > 10:
        raise TooBig()
    return n

def outer(n: int) -> int:
    return check(n) + 1

try:
    print(outer(5))
    print(outer(50))
except TooBig:
    print("too big")
