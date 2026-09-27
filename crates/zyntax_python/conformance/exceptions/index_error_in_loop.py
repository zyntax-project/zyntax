import math


def total(data, n):
    s = 0.0
    for i in range(n):
        s += math.sqrt(data[i])
    return s


def guarded(data, n):
    log = []
    try:
        try:
            return total(data, n)
        finally:
            log.append("finally")
    except IndexError as e:
        log.append("caught " + str(e))
    return log


def gen(data):
    for i in range(len(data) + 1):
        yield data[i]


def consume(g):
    s = 0
    for v in g:
        s += v
    return s


def run(data):
    try:
        return consume(gen(data))
    except IndexError:
        return -1


def main():
    data = [float(i) for i in range(10)]
    print(round(guarded(data, 10), 6))
    print(guarded(data, 11))
    for k in range(3):
        print(round(total(data, 10), 6), guarded(data, 12))
    print(run([1, 2, 3]))
    try:
        total([4.0, -1.0], 2)
    except ValueError:
        print("ValueError")


main()
