# The n-body simulation from the Computer Language Benchmarks Game:
# five bodies, ten million steps of symplectic Euler, then the energy.
# Instances with float fields in a list, read and written in the inner
# pair loop.
# Returns -169077: the energy times a million, truncated.

import math

class Body:
    def __init__(self, x: float, y: float, z: float, vx: float, vy: float, vz: float, mass: float):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass

def advance(bodies: list[Body], n: int, dt: float) -> int:
    for i in range(n):
        a = bodies[i]
        for j in range(i + 1, n):
            b = bodies[j]
            dx = a.x - b.x
            dy = a.y - b.y
            dz = a.z - b.z
            d2 = dx * dx + dy * dy + dz * dz
            distance = math.sqrt(d2)
            mag = dt / (distance * distance * distance)
            a.vx = a.vx - dx * b.mass * mag
            a.vy = a.vy - dy * b.mass * mag
            a.vz = a.vz - dz * b.mass * mag
            b.vx = b.vx + dx * a.mass * mag
            b.vy = b.vy + dy * a.mass * mag
            b.vz = b.vz + dz * a.mass * mag
    for k in range(n):
        body = bodies[k]
        body.x = body.x + dt * body.vx
        body.y = body.y + dt * body.vy
        body.z = body.z + dt * body.vz
    return n

def energy(bodies: list[Body], n: int) -> float:
    e = 0.0
    for i in range(n):
        a = bodies[i]
        e = e + 0.5 * a.mass * (a.vx * a.vx + a.vy * a.vy + a.vz * a.vz)
        for j in range(i + 1, n):
            b = bodies[j]
            dx = b.x - a.x
            dy = b.y - a.y
            dz = b.z - a.z
            d2 = dx * dx + dy * dy + dz * dz
            e = e - (a.mass * b.mass) / math.sqrt(d2)
    return e

def main() -> int:
    solar_mass = 39.478417604357434
    days_per_year = 365.24
    sun = Body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, solar_mass)
    jupiter = Body(
        4.84143144246472, -1.16032004402742, -0.103622044471123,
        0.00166007664274403 * days_per_year,
        0.00769901118419740 * days_per_year,
        -0.0000690460016972 * days_per_year,
        0.000954791938424 * solar_mass,
    )
    saturn = Body(
        8.34336671824458, 4.12479856412430, -0.403523417114321,
        -0.00276742510726862 * days_per_year,
        0.00499852801234917 * days_per_year,
        0.0000230417297573 * days_per_year,
        0.000285885980666 * solar_mass,
    )
    uranus = Body(
        12.8943695621391, -15.1111514016986, -0.223307578892655,
        0.00296460137564761 * days_per_year,
        0.00237847173959480 * days_per_year,
        -0.0000296589568540 * days_per_year,
        0.0000436624404335 * solar_mass,
    )
    neptune = Body(
        15.3796971148509, -25.9193146099879, 0.179258772950371,
        0.00268067772490389 * days_per_year,
        0.00162824170038242 * days_per_year,
        -0.0000951592254519 * days_per_year,
        0.0000515138902046 * solar_mass,
    )
    bodies = [sun, jupiter, saturn, uranus, neptune]
    n = 5
    px = 0.0
    py = 0.0
    pz = 0.0
    for b in bodies:
        px = px + b.vx * b.mass
        py = py + b.vy * b.mass
        pz = pz + b.vz * b.mass
    sun.vx = 0.0 - px / solar_mass
    sun.vy = 0.0 - py / solar_mass
    sun.vz = 0.0 - pz / solar_mass
    dt = 0.01
    for outer in range(20):
        for step in range(500000):
            advance(bodies, n, dt)
    e = energy(bodies, n)
    return int(e * 1000000.0)

print(main())
