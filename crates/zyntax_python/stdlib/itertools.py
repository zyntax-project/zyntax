# The functions of the standard `itertools` module that a program reaches
# by name. The endless ones are iterator classes, pulled one item at a
# time. The others build their items from what they are given and return
# an iterator over them, so they take finite input.


class _ListIter:
    def __init__(self, items):
        self.items = items
        self.at = 0

    def __next__(self):
        if self.at >= len(self.items):
            raise StopIteration
        item = self.items[self.at]
        self.at += 1
        return item


class _Source:
    """What an itertools function reads its input through: an iterator
    object is pulled one item at a time, anything else is listed."""

    def __init__(self, iterable):
        self.lazy = hasattr(iterable, "__next__")
        self.source = iterable
        self.items = []
        if not self.lazy:
            self.items = list(iterable)
        self.at = 0
        self.value = None

    def advance(self):
        if self.lazy:
            try:
                self.value = self.source.__next__()
            except StopIteration:
                return False
            return True
        if self.at >= len(self.items):
            return False
        self.value = self.items[self.at]
        self.at += 1
        return True


class count:
    def __init__(self, start=0, step=1):
        self.n = start
        self.step = step

    def __next__(self):
        value = self.n
        self.n = self.n + self.step
        return value


class cycle:
    def __init__(self, iterable):
        self.saved = list(iterable)
        self.at = 0

    def __next__(self):
        if len(self.saved) == 0:
            raise StopIteration
        value = self.saved[self.at % len(self.saved)]
        self.at += 1
        return value


class repeat:
    def __init__(self, obj, times=None):
        self.obj = obj
        self.forever = times is None
        self.left = 0
        if times is not None:
            self.left = times

    def __next__(self):
        if not self.forever:
            if self.left <= 0:
                raise StopIteration
            self.left -= 1
        return self.obj


def chain(*iterables):
    out = []
    for it in iterables:
        for item in it:
            out.append(item)
    return _ListIter(out)


def chain_from_iterable(iterables):
    out = []
    for it in iterables:
        for item in it:
            out.append(item)
    return _ListIter(out)


def islice(iterable, *bounds):
    start = 0
    step = 1
    stop = None
    if len(bounds) == 1:
        stop = bounds[0]
    elif len(bounds) == 2:
        start = bounds[0]
        stop = bounds[1]
    else:
        start = bounds[0]
        stop = bounds[1]
        step = bounds[2]
    if start is None:
        start = 0
    if step is None:
        step = 1
    out = []
    src = _Source(iterable)
    index = 0
    wanted = start
    while stop is None or index < stop:
        if not src.advance():
            break
        if index == wanted:
            out.append(src.value)
            wanted += step
        index += 1
    return _ListIter(out)


def accumulate(iterable, func=None, initial=None):
    out = []
    total = initial
    first = initial is None
    if initial is not None:
        out.append(initial)
    for item in iterable:
        if first:
            total = item
            first = False
        elif func is None:
            total = total + item
        else:
            total = func(total, item)
        out.append(total)
    return _ListIter(out)


def takewhile(predicate, iterable):
    out = []
    src = _Source(iterable)
    while src.advance():
        if predicate(src.value):
            out.append(src.value)
        else:
            break
    return _ListIter(out)


def dropwhile(predicate, iterable):
    out = []
    dropping = True
    for item in iterable:
        if dropping and predicate(item):
            continue
        dropping = False
        out.append(item)
    return _ListIter(out)


def filterfalse(predicate, iterable):
    out = []
    for item in iterable:
        if predicate is None:
            if not item:
                out.append(item)
        elif not predicate(item):
            out.append(item)
    return _ListIter(out)


def compress(data, selectors):
    chosen = list(selectors)
    out = []
    i = 0
    for item in data:
        if i >= len(chosen):
            break
        if chosen[i]:
            out.append(item)
        i += 1
    return _ListIter(out)


def starmap(func, iterable):
    out = []
    for args in iterable:
        n = len(args)
        if n == 0:
            out.append(func())
        elif n == 1:
            out.append(func(args[0]))
        elif n == 2:
            out.append(func(args[0], args[1]))
        elif n == 3:
            out.append(func(args[0], args[1], args[2]))
        else:
            out.append(func(args[0], args[1], args[2], args[3]))
    return _ListIter(out)


def zip_longest(*iterables, fillvalue=None):
    lists = []
    for it in iterables:
        lists.append(list(it))
    longest = 0
    for items in lists:
        if len(items) > longest:
            longest = len(items)
    out = []
    i = 0
    while i < longest:
        row = []
        for items in lists:
            if i < len(items):
                row.append(items[i])
            else:
                row.append(fillvalue)
        out.append(tuple(row))
        i += 1
    return _ListIter(out)


def pairwise(iterable):
    items = list(iterable)
    out = []
    i = 1
    while i < len(items):
        out.append((items[i - 1], items[i]))
        i += 1
    return _ListIter(out)


def batched(iterable, n):
    out = []
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            out.append(tuple(batch))
            batch = []
    if len(batch) > 0:
        out.append(tuple(batch))
    return _ListIter(out)


def product(*iterables, repeat=1):
    pools = []
    r = 0
    while r < repeat:
        for it in iterables:
            pools.append(list(it))
        r += 1
    out = []
    if len(pools) == 0:
        out.append(())
        return _ListIter(out)
    for pool in pools:
        if len(pool) == 0:
            return _ListIter(out)
    positions = [0] * len(pools)
    while True:
        row = []
        k = 0
        while k < len(pools):
            row.append(pools[k][positions[k]])
            k += 1
        out.append(tuple(row))
        k = len(pools) - 1
        while k >= 0:
            positions[k] += 1
            if positions[k] < len(pools[k]):
                break
            positions[k] = 0
            k -= 1
        if k < 0:
            return _ListIter(out)


def permutations(iterable, r=None):
    pool = list(iterable)
    n = len(pool)
    if r is None:
        r = n
    out = []
    if r > n:
        return _ListIter(out)
    indices = list(range(n))
    cycles = list(range(n, n - r, -1))
    first = []
    for i in range(r):
        first.append(pool[indices[i]])
    out.append(tuple(first))
    while n > 0:
        i = r - 1
        found = False
        while i >= 0:
            cycles[i] -= 1
            if cycles[i] == 0:
                moved = indices[i]
                del indices[i]
                indices.append(moved)
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[n - j] = indices[n - j], indices[i]
                row = []
                for k in range(r):
                    row.append(pool[indices[k]])
                out.append(tuple(row))
                found = True
                break
            i -= 1
        if not found:
            return _ListIter(out)
    return _ListIter(out)


def combinations(iterable, r):
    pool = list(iterable)
    n = len(pool)
    out = []
    if r > n:
        return _ListIter(out)
    indices = list(range(r))
    first = []
    for i in indices:
        first.append(pool[i])
    out.append(tuple(first))
    while True:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            return _ListIter(out)
        indices[i] += 1
        j = i + 1
        while j < r:
            indices[j] = indices[j - 1] + 1
            j += 1
        row = []
        for k in indices:
            row.append(pool[k])
        out.append(tuple(row))


def combinations_with_replacement(iterable, r):
    pool = list(iterable)
    n = len(pool)
    out = []
    if n == 0 and r > 0:
        return _ListIter(out)
    indices = [0] * r
    first = []
    for i in indices:
        first.append(pool[i])
    out.append(tuple(first))
    while True:
        i = r - 1
        while i >= 0 and indices[i] == n - 1:
            i -= 1
        if i < 0:
            return _ListIter(out)
        value = indices[i] + 1
        j = i
        while j < r:
            indices[j] = value
            j += 1
        row = []
        for k in indices:
            row.append(pool[k])
        out.append(tuple(row))


def groupby(iterable, key=None):
    out = []
    current = None
    group = []
    started = False
    for item in iterable:
        k = item
        if key is not None:
            k = key(item)
        if started and k == current:
            group.append(item)
        else:
            if started:
                out.append((current, group))
            current = k
            group = [item]
            started = True
    if started:
        out.append((current, group))
    return _ListIter(out)


def tee(iterable, n=2):
    items = list(iterable)
    out = []
    for i in range(n):
        out.append(_ListIter(list(items)))
    return tuple(out)
