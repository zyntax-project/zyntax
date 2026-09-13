# Recursive calls: fib(40), the canonical call-dispatch kernel.
# Returns 102334155.

def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def main() -> int:
    return fib(40)

print(main())
