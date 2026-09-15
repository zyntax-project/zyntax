one = {"y": [1, 2]}
one["y"].append(3)
print(one)

items = [1, 2]
three = {"z": items}
three["z"].append(3)
print(three, items)

outer = [items]
outer[0].append(4)
print(outer, items, three)

class Holder:
    def __init__(self):
        self.items = [1, 2]

holder = Holder()
holder.items.append(3)
other = holder.items
other.append(4)
print(holder.items, other)

for i in range(10):
    one["y"].append(i)
print(one["y"])
