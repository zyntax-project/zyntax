class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

    def fail(self):
        self.value += 10
        raise ValueError("failed")

counter = Counter()
objects = [None]
objects[0] = counter
for obj in objects:
    print(obj.increment())
    print(obj.value)
    try:
        obj.fail()
    except ValueError:
        print(obj.value)

values = [counter.increment(), counter.increment()]
print(values, counter.value)

def no_result():
    counter.increment()

print([no_result()], counter.value)
