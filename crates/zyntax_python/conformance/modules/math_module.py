# the math module: functions and constants, imported whole and by name
import math
from math import sqrt, pi, floor

print(math.sqrt(16), sqrt(2.25))
print(math.floor(2.7), math.ceil(2.1), floor(-2.5), math.ceil(-2.5))
print(math.trunc(3.9), math.trunc(-3.9))
print(math.fabs(-3.5), math.pow(2, 10))
print(round(math.pi, 5), round(pi, 2), round(math.e, 3), round(math.tau, 4))
print(math.exp(0), math.log(1), math.log(8, 2), math.log2(1024), math.log10(1000))
print(round(math.sin(0), 3), round(math.cos(0), 3), round(math.sin(math.pi / 2), 3))
print(round(math.atan2(1, 1), 4), round(math.hypot(3, 4), 1))
print(math.gcd(12, 18), math.gcd(7, 0), math.gcd(-4, 6))
print(math.factorial(5), math.factorial(0), math.factorial(10))
print(math.isnan(float("nan")), math.isinf(math.inf), math.isinf(-math.inf), math.isfinite(1.5))
print(math.inf > 10 ** 18, -math.inf < 0)
print(round(math.degrees(math.pi), 1), round(math.radians(180), 4))
print(math.fmod(7.5, 2), math.copysign(3, -1))
print(math.floor(7), math.sqrt(9) == 3)

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

print(dist(0, 0, 3, 4))
values = [1.0, 4.0, 9.0]
print([sqrt(v) for v in values])
print(sum(math.floor(v / 2) for v in values))
