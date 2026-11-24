from collections import Counter, OrderedDict, defaultdict, deque

groups = defaultdict(list)
for word in ["apple", "avocado", "banana", "blueberry", "cherry"]:
    groups[word[0]].append(word)
print(sorted(groups.items()), len(groups), "a" in groups, "z" in groups)
print(dict(groups), groups["a"], groups["zzz"], len(groups))
print(sorted(groups), list(groups.keys()) == list(groups), sorted(groups.values())[0])

counts = defaultdict(int)
for ch in "mississippi":
    counts[ch] += 1
print(sorted(counts.items()), sum(counts.values()))
nested = defaultdict(lambda: defaultdict(int))
nested["x"]["y"] += 2
nested["x"]["z"] += 3
print(sorted(nested["x"].items()), len(nested))
seen = defaultdict(set)
seen[1].add("a")
seen[1].add("b")
print(sorted(seen[1]))
plain = defaultdict()
try:
    plain["missing"]
except KeyError as e:
    print("KeyError", e)
print(counts.get("m"), counts.get("q", -1), counts.pop("m"), "m" in counts)

c = Counter("abracadabra")
print(c.most_common(2), c["a"], c["z"], len(c), sum(c.values()), c.total())
print(sorted(c.elements()), dict(c.most_common()))
c.update("aaa")
c.subtract("b")
print(c["a"], c["b"], c.most_common(1))
d = Counter({"x": 3, "y": 1})
print(sorted((c + d).items()), sorted((d - Counter({"x": 5})).items()))
print(Counter([1, 1, 2, 3, 3, 3]).most_common(), Counter().most_common())
words = "the quick brown the lazy the end quick".split()
top = Counter(words)
print(top.most_common(2), [w for w, n in top.items() if n > 1])

q = deque([1, 2, 3])
q.append(4)
q.appendleft(0)
print(q.popleft(), q.pop(), list(q), len(q), q[0], q[-1], 2 in q, 9 in q)
q.extend([7, 8])
q.extendleft([-1, -2])
print(list(q))
q.rotate(2)
print(list(q))
q.rotate(-3)
print(list(q), q.count(7), q.index(8))
q.reverse()
print(list(q))
q.remove(7)
q.clear()
print(list(q), len(q), bool(q))
window = deque(maxlen=3)
for i in range(6):
    window.append(i)
print(list(window), window.maxlen, sum(window))
left = deque(maxlen=2)
left.appendleft(1)
left.appendleft(2)
left.appendleft(3)
print(list(left))
try:
    deque().pop()
except IndexError as e:
    print("IndexError", e)
try:
    deque().popleft()
except IndexError as e:
    print("IndexError", e)
queue = deque()
for i in range(100):
    queue.append(i)
total = 0
while queue:
    total += queue.popleft()
    if total % 7 == 0 and len(queue) < 50:
        queue.append(1)
print(total)

od = OrderedDict()
od["b"] = 1
od["a"] = 2
od["c"] = 3
od.move_to_end("b")
print(list(od.keys()), list(od.items()), od.popitem(), od.popitem(last=False))
