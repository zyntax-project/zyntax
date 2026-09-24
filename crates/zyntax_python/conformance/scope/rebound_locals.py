# a local rebound in straight-line code is a fresh variable from there on;
# one a loop, a merge or a later-running body reads keeps one binding
from array import array


def encrypt(data):
    # the parameter is bytes; from here on data is an array of them
    data = array('B', data)
    for i in range(len(data)):
        data[i] ^= 0x5A
    return data.tobytes()


def widen(n):
    total = 0
    for i in range(n):
        total += i
    total = total / 2
    return total


def by_branch(x, flag):
    if flag:
        x = str(x)
    return x


def early(x, flag):
    if flag:
        x = [x]
        return len(x)
    return x + 1


def genexp_iterable(xs):
    g = (v * 10 for v in xs)
    xs = [7, 8]
    return list(g), xs


def genexp_inner(xs, k):
    g = (v * k for v in xs)
    k = 100
    return list(g)


def lambda_reads(x):
    k = lambda: x
    x = x * 2
    return k(), x


def nested_reads(x):
    def k():
        return x
    x = x * 2
    return k(), x


def loop_carried(n):
    x = 1
    for _ in range(n):
        x = x * 2
    return x


def while_carried(n):
    x = n
    steps = 0
    while x != 1:
        x = x // 2 if x % 2 == 0 else 3 * x + 1
        steps += 1
    return steps


def twice(s):
    s = s.split(",")
    s = len(s)
    return s


def comprehension_reads(x):
    x = x + 1
    squares = [x * y for y in range(3)]
    x = str(x)
    return squares, x


def with_body(x):
    with open("/dev/null", "rb") as f:
        x = [x, len(f.read())]
        return x


print(encrypt(b"hello"))
print(widen(5))
print(by_branch(3, True), by_branch(3, False))
print(early(3, True), early(3, False))
print(genexp_iterable([1, 2, 3]))
print(genexp_inner([1, 2, 3], 2))
print(lambda_reads(4))
print(nested_reads(4))
print(loop_carried(5), while_carried(6))
print(twice("a,b,c"))
print(comprehension_reads(2))
print(with_body(1))
