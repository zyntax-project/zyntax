# A literal's keys meet by value: a bool or an integral float equal to
# an int key is that key, the last value written wins and the first key
# stays; distinct keys keep their order.
d = {1: 'a', True: 'b', 1.0: 'c'}
print(d, len(d))
e = {'x': 1, 'y': 2, 'x': 3}
print(e, len(e))
f = {-1: 'm', 1: 'p', 0: 'z'}
print(f, f[-1], f[0])
g = {2: 'two', 3: 'three', 5: 'five', 7: 'seven', 11: 'eleven',
     13: 'thirteen', 17: 'seventeen', 19: 'nineteen', 23: 'twenty-three'}
print(len(g), g[23], 4 in g, 23.0 in g, list(g)[:3])
h = {'a': [1], 'b': [2, 3], 'c': []}
h['c'].append(4)
print(h, sum(len(v) for v in h.values()))
k = {0.5: 'half', 2.0: 'two', 2: 'int two'}
print(k, len(k))
t = {(1, 2): 'p', (1.0, 2.0): 'q'}
print(t, len(t))
