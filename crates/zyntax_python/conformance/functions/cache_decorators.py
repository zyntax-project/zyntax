import functools
from functools import lru_cache, cache

calls = []


@lru_cache(maxsize=None)
def fib(n):
    calls.append(n)
    return n if n < 2 else fib(n - 1) + fib(n - 2)


print(fib(30), len(calls))


@cache
def grid_paths(rows, cols):
    if rows == 1 or cols == 1:
        return 1
    return grid_paths(rows - 1, cols) + grid_paths(rows, cols - 1)


print(grid_paths(10, 10), grid_paths(3, 4))


@functools.lru_cache(maxsize=128)
def join(prefix, word="x", *, sep="-"):
    calls.append(len(prefix))
    return prefix + sep + word


before = len(calls)
print(join("a"), join("a"), join("a", "b"), join("a", word="c"), join("a", sep="+"))
print(len(calls) - before)


@lru_cache
def square(x):
    return x * x


print([square(i) for i in range(5)], square(4))
