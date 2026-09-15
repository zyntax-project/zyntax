# An empty list literal takes the element kind its binding's annotation
# names, so the pieces appended to it are stored and joined as that kind.

def build() -> str:
    parts: list[str] = []
    for i in range(5):
        parts.append("p" + str(i))
    return ",".join(parts)


def total() -> int:
    nums: list[int] = []
    for i in range(5):
        nums.append(i * i)
    return sum(nums)


def mixed() -> str:
    anything: list = []
    anything.append(1)
    anything.append("two")
    return str(anything)


print(build())
print(total())
print(mixed())
