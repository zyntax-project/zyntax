# Dicts and sets whose keys and values settle to one kind each hold them
# as that kind; what reaches them dynamically, a key of another kind,
# an alias and a callee that writes another kind all see one storage.
def first():
    d = {'a': 1, 'b': 2}
    d['c'] = 3
    print(d, len(d), d['a'], d.get('z'), d.get('b', 0), 'c' in d, 'q' in d, 1 in d)
    n = {1: 'x', 2: 'y'}
    n[3] = 'z'
    print(n, n[2], n.get(5, 'none'), 2.0 in n, True in n, n.get(1.0), n.pop(3), n)
    del n[1]
    print(n, list(n), list(n.values()), sorted(n.items()))
    t = {(1, 2): 'a'}
    print((1, 2) in t, (1, 2.0) in t, t.get((1, 2.0)))
    c = {}
    for w in 'the cat the dog the end'.split():
        c[w] = c.get(w, 0) + 1
    print(c, sorted(c.items()))
    s = set()
    for i in range(20):
        s.add(i % 7)
    print(sorted(s), 3 in s, 3.0 in s, 9 in s)
    print({1: 2} == {1.0: 2.0}, {1: 2} == {1: 3}, n == {2: 'y'})
    fs = [frozenset([1, 2]), frozenset([3])]
    print(fs, fs[0] | fs[1], len({fs[0], frozenset([2, 1])}))
    o = [s, d]
    print(o[0] == s, 5 in o[0], o[1]['a'])
    x = dict((i, i * i) for i in range(4))
    print(x, x[3])
first()

import json

def dyn(o, k):
    print(len(o), k in o, sorted(o), o[k] if k in o else None)

def dyn_set(a: object, b: object):
    print(sorted(a - b), sorted(a | b), sorted(a & b), sorted(a ^ b), a <= b, a < b, b >= a)

def grow(m):
    m['c'] = 'q'

def main():
    d = {'a': 1, 'b': 2}
    e = d
    e['z'] = 26
    print(d, e is d)
    lists = {'x': [1, 2]}
    lists['x'].append(3)
    lists['y'] = [4]
    print(lists, lists['x'][2], len(lists['y']))
    nested = {'p': {'q': 1}}
    nested['p']['r'] = 2
    inner = nested['p']
    inner['s'] = 3
    print(nested)
    dyn(d, 'a')
    dyn(d, 'nope')
    g = {'a': 1}
    grow(g)
    print(g)
    print(json.dumps({'k': 1, 'j': 2}), json.dumps({'n': [1, 2]}))
    s = {1, 2, 3}
    t = frozenset([2, 3, 4])
    dyn_set(s, t)
    dyn_set(t, s)
    dyn_set(t, frozenset([2.0, 3.0, 4.0, 5.0]))
    print([t], {'t': t}, (t, 1))
    q = {i: i * i for i in range(5)}
    print(q.pop(2), q.pop(9, -1), q.setdefault(7, 49), q.setdefault(1, 0), q)
    q.update({10: 100})
    c = q.copy()
    c[11] = 121
    print(len(q), len(c), q.popitem(), q)
    c.clear()
    print(c, len(c), bool(c), bool(q))
    names = {x for x in 'hello world' if x != ' '}
    print(sorted(names), 'h' in names, 1 in names)
    print(sorted(d), list(d.keys()), max(d), min(d.values()), sum(d.values()))
    total = 0
    for k, v in d.items():
        total += v
    for k in d:
        total += len(k)
    for v in d.values():
        total += v
    print(total)
    pairs = {(i, j): i + j for i in range(3) for j in range(3)}
    print(pairs[(1, 2)], (1, 2.0) in pairs, pairs.get((2, 2)), (5, 5) in pairs)
    fl = {0.5: 'h', 1.0: 'o'}
    print(fl[1], 1 in fl, fl.get(True), 2 in fl)
    b = {'x': True, 'y': False}
    b['z'] = True
    print(b, b['x'], sum(1 for v in b.values() if v))
    objs = {}
    objs['k'] = None
    objs['j'] = 1
    print(objs)
    print(d == {'a': 1, 'b': 2, 'z': 26}, d != {'a': 1}, {1: 1.0} == {1: 1}, s == {1.0, 2.0, 3.0}, t == frozenset([4, 3, 2]))

main()


def add_c(m):
    m['c'] = 'q'


def aliases():
    d = {'a': 1}
    add_c(d)
    print(d, d['a'])
    d2 = {'a': 1}
    e = d2
    e['b'] = 'x'
    print(d2)
    x = {}
    y = x
    y = {1: 2}
    z = x
    z[5] = 6
    print(x, y)
    s = {1, 2}
    t = s
    t.add('three')
    print(sorted(s, key=str))


aliases()
