# A name nothing binds raises NameError where it is read, and a name a
# standard-library module lacks raises AttributeError.
import math


def f():
    return time.time()


try:
    f()
except NameError as e:
    print("NameError", e)
try:
    print(nosuch)
except NameError as e:
    print("NameError", e)


def g():
    return missing_name + 1


try:
    g()
except NameError as e:
    print("NameError", e)
print("after")
try:
    print(math.nope)
except AttributeError as e:
    print("AttributeError", e)
if False:
    print(never_read)


def h():
    return later


later = 5
print(h(), math.pi > 3)
