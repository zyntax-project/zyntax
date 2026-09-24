# A loop whose bound is a constant, a million times round: the function
# is compiled before its first call, which runs its native code.
class Bag:
    def __init__(self, payload):
        self.payload = payload


def main() -> int:
    total = 0.0
    for i in range(1000000):
        bag = Bag(1.5)
        total = total + bag.payload
    return int(total)


print(main())
