import sys


def main(n):
    l = []
    for i in range(n):
        t0 = time.time()
        l.append(time.time() - t0 >= 0)
    return l


if __name__ == '__main__':
    import lib.text, math, time
    print(main(2), math.sqrt(16), lib.text.banner("hi"))
