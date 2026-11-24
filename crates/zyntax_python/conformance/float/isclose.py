import math
from math import isclose
print(math.isclose(0.1 + 0.2, 0.3), math.isclose(1.0, 1.1))
print(math.isclose(1.0, 1.05, rel_tol=0.1), math.isclose(0.0, 1e-10, abs_tol=1e-9))
print(math.isclose(0.0, 1e-10), math.isclose(1e10, 1e10 + 1, rel_tol=1e-9))
print(isclose(1, 1), isclose(2, 3), isclose(5, 5.000001, abs_tol=0.001))
print(isclose(float("inf"), float("inf")), isclose(1.0, float("inf")))
print(isclose(float("-inf"), float("inf")), isclose(float("nan"), float("nan")))
print(isclose(a=1.0, b=1.0 + 1e-12))
