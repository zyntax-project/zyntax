# A driver loop whose body binds the None a call returns, so a loop
# value of no type crosses its header, in a function that returns the
# list it fills. The interpreted frame leaves into the region outlined
# for it once the optimizing tier has compiled the region, and the
# parameters after that value must arrive where the region reads them.
def work(k, acc):
    s = acc[0]
    for j in range(k):
        s = (s * 31 + j) % 1000003
    acc[0] = s


def main(n):
    acc = [1]
    times = []
    for i in range(n):
        o = work(20000000 if i >= 60 else 1000, acc)
        times.append(1.0 * i)
    times.append(1.0 * acc[0])
    return times


def run(n, f):
    data = f(n)
    total = 0.0
    for x in data:
        total += x
    print(len(data), total)


run(70, main)
