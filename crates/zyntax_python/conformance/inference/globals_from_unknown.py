# A global built by a function from another global: its type is
# decided once that function's parameter is, not from the round that
# saw the parameter undecided. The default of a parameter no call
# passes is that global.
def combinations(l):
    result = []
    ll = list(l)
    for x in range(len(ll) - 1):
        ls = ll[x + 1:]
        for y in ls:
            result.append((ll[x], y))
    return result


BODIES = {
    "a": ([1.0, 2.0], [0.5, 0.25], 3.0),
    "b": ([4.0, 5.0], [0.125, 0.0625], 6.0),
    "c": ([7.0, 8.0], [1.0, 2.0], 9.0),
}
SYSTEM = list(BODIES.values())
PAIRS = combinations(SYSTEM)
FIRST, SECOND = PAIRS[0]
COUNT = len(PAIRS)


def advance(dt, n, bodies=SYSTEM, pairs=PAIRS):
    for i in range(n):
        for (([x1, y1], v1, m1), ([x2, y2], v2, m2)) in pairs:
            dx = x1 - x2
            v1[0] -= dx * m2 * dt
            v2[0] += dx * m1 * dt
        for (r, [vx, vy], m) in bodies:
            r[0] += dt * vx


def energy(bodies=SYSTEM, pairs=PAIRS, e=0.0):
    for (((x1, y1), v1, m1), ((x2, y2), v2, m2)) in pairs:
        e -= (m1 * m2) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    for (r, [vx, vy], m) in bodies:
        e += m * (vx * vx + vy * vy) / 2.0
    return e


print(COUNT, FIRST[2], SECOND[2])
print(round(energy(), 6))
advance(0.01, 10)
print(round(energy(), 6))
print([round(v, 6) for v in SYSTEM[0][0]], [round(v, 6) for v in SYSTEM[2][1]])
