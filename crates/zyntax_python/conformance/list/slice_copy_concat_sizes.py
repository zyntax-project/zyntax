# Copies, slices and joins of every small length, around the size of a
# list's first buffer, of ints, strings and nested lists.
for n in range(0, 11):
    xs = list(range(n))
    ss = [str(i) + "s" for i in range(n)]
    nested = [[i] for i in range(n)]
    print(n, xs[:], xs + xs, xs[1:], xs[:-1], xs[-3:], xs[::2], xs[::-1], xs[2:7], xs[n:], xs[5:2])
    print(ss[:], ss + ss[:2], ss[1:-1])
    print(nested[:], nested[1:] + nested[:1])
    copy = xs[:]
    copy.append(99)
    print(len(xs), len(copy), copy[-1], xs == copy[:-1])
    ys = xs[:]
    ys[1:] = xs[2:] + xs[1:2]
    print(ys)
big = list(range(1000))
print(len(big[:]), len(big[10:990]), sum(big[100:200] + big[:50]))
total = 0
for k in range(2000):
    part = big[k % 7:k % 7 + 6]
    total += len(part) + part[0]
print(total)
