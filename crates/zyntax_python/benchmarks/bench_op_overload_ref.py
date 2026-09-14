# Operator overloading: `+` on a three-float class, two million
# times, each producing a new instance.
# Returns 21000000.

class Vec3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

def main() -> int:
    a = Vec3(1.0, 2.0, 3.0)
    b = Vec3(4.0, 5.0, 6.0)
    acc = Vec3(0.0, 0.0, 0.0)
    for i in range(1000000):
        acc = acc + a
        acc = acc + b
    return int(acc.x + acc.y + acc.z)

print(main())
