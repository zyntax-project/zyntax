import re


def size_of(data):
    if (n := len(data)) > 3:
        return "long " + str(n)
    elif (m := n * 2) > 2:
        return "medium " + str(m)
    else:
        return "short " + str(n)


print(size_of([1, 2, 3, 4]), size_of([1, 2]), size_of([1]))

data = [1, 2, 3]
if (count := len(data)) > 2:
    print("count", count)
print((total := sum(data)) + 1, total)
half = (total // 2)
print(half)

values = iter([3, 2, 1, 0, 5])
steps = 0
while (item := next(values, None)) is not None and item > 0:
    steps += item
print(steps, item)

queue = [4, 5, 6]
seen = []
while (head := queue.pop(0) if queue else None) is not None:
    seen.append(head)
print(seen)

words = ["a", "bb", "ccc"]
lengths = []
i = 0
while (i := i + 1) <= len(words):
    lengths.append(len(words[i - 1]))
print(lengths, i)
if (found := "needle" in "haystack with needle") and found:
    print("found", found)
