# Method call latency on a class instance, against
# `bench_inlined_call.py`. The receiver is statically the class, so
# the call is direct and the field access typed; the delta to the
# baseline is what the method shape leaves after inlining.
# Returns 350000000.

class Acc:
    def __init__(self) -> None:
        self.total = 0

    def step(self, i: int) -> None:
        self.total = self.total + (i % 8)

def main() -> int:
    acc = Acc()
    i = 0
    while i < 100000000:
        acc.step(i)
        i += 1
    return acc.total

print(main())
