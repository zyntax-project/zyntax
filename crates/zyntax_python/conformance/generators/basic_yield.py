# test_generators: yield, iteration, exhaustion
def count_up(n: int):
    i = 0
    while i < n:
        yield i
        i += 1

for v in count_up(4):
    print(v)
print(list(count_up(3)))
print(sum(count_up(101)))
