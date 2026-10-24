# A loop over reversed(range(...)) counts down, reads each bound once and
# in order, and leaves its target as a loop over the list would.
trace = []


def bound(v):
    trace.append(v)
    return v


def down(n):
    out = []
    for i in reversed(range(n)):
        out.append(i)
    return out


def down_between(a, b):
    out = []
    for i in reversed(range(a, b)):
        out.append(i)
    return out


print(down(5), down(1), down(0), down(-3))
print(down_between(2, 7), down_between(-3, 2), down_between(5, 2), down_between(4, 4))

for i in reversed(range(bound(1), bound(4))):
    trace.append(i)
print(trace)

i = 99
for i in reversed(range(0)):
    pass
print(i)
for i in reversed(range(3)):
    pass
print(i)
for i in reversed(range(10)):
    if i == 6:
        break
print(i)

total = 0
for i in reversed(range(4)):
    for j in reversed(range(i)):
        total += i * 10 + j
print(total)

big = 0
for k in reversed(range(1, 100001)):
    big += k
print(big)
