# The containers of the standard `collections` module that a program
# reaches by name. Each keeps its elements in a dict or a list, so a value
# of one is read through the methods below, not as a builtin.


class _Missing:
    pass


_MISSING = _Missing()


class _Dict:
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return len(self.data) > 0

    def __iter__(self):
        for key in self.data:
            yield key

    def __eq__(self, other):
        if isinstance(other, _Dict):
            return self.data == other.data
        return self.data == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def keys(self):
        return list(self.data.keys())

    def values(self):
        return list(self.data.values())

    def items(self):
        return list(self.data.items())

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default

    def pop(self, key, default=_MISSING):
        if key in self.data:
            return self.data.pop(key)
        if default is _MISSING:
            raise KeyError(key)
        return default

    def popitem(self, last=True):
        if len(self.data) == 0:
            raise KeyError("dictionary is empty")
        keys = list(self.data.keys())
        key = keys[len(keys) - 1]
        if not last:
            key = keys[0]
        value = self.data.pop(key)
        return key, value

    def setdefault(self, key, default=None):
        if key not in self.data:
            self.data[key] = default
        return self.data[key]

    def update(self, other):
        if isinstance(other, _Dict):
            for key in other.data:
                self.data[key] = other.data[key]
        else:
            self.data.update(other)

    def clear(self):
        self.data.clear()

    def to_dict(self):
        return dict(self.data)


class defaultdict(_Dict):
    def __init__(self, factory=None, initial=None):
        self.data = {}
        self.factory = factory
        if initial is not None:
            self.update(initial)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        factory = self.factory
        if factory is None:
            raise KeyError(key)
        value = factory()
        self.data[key] = value
        return value

    def __missing__(self, key):
        return self.__getitem__(key)

    def copy(self):
        return defaultdict(self.factory, self.data)

    def __repr__(self):
        return "defaultdict(" + str(self.factory) + ", " + repr(self.data) + ")"


class OrderedDict(_Dict):
    def __init__(self, initial=None):
        self.data = {}
        if initial is not None:
            self.update(initial)

    def move_to_end(self, key, last=True):
        value = self.data.pop(key)
        if last:
            self.data[key] = value
        else:
            rest = list(self.data.items())
            self.data.clear()
            self.data[key] = value
            for k, v in rest:
                self.data[k] = v

    def copy(self):
        return OrderedDict(self.data)

    def __repr__(self):
        if len(self.data) == 0:
            return "OrderedDict()"
        return "OrderedDict(" + repr(self.data) + ")"


class Counter(_Dict):
    def __init__(self, iterable=None):
        self.data = {}
        if iterable is not None:
            self.update(iterable)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return 0

    def update(self, other):
        if isinstance(other, _Dict):
            for key in other.data:
                self.data[key] = self.data.get(key, 0) + other.data[key]
        elif isinstance(other, dict):
            for key in other:
                self.data[key] = self.data.get(key, 0) + other[key]
        else:
            for item in other:
                self.data[item] = self.data.get(item, 0) + 1

    def subtract(self, other):
        if isinstance(other, _Dict):
            for key in other.data:
                self.data[key] = self.data.get(key, 0) - other.data[key]
        elif isinstance(other, dict):
            for key in other:
                self.data[key] = self.data.get(key, 0) - other[key]
        else:
            for item in other:
                self.data[item] = self.data.get(item, 0) - 1

    def most_common(self, n=None):
        ordered = sorted(self.data.items(), key=lambda pair: -pair[1])
        if n is None:
            return ordered
        return ordered[:n]

    def elements(self):
        out = []
        for key in self.data:
            count = self.data[key]
            i = 0
            while i < count:
                out.append(key)
                i += 1
        return out

    def total(self):
        t = 0
        for key in self.data:
            t += self.data[key]
        return t

    def copy(self):
        return Counter(self.data)

    def __add__(self, other):
        out = Counter()
        for key in self.data:
            total = self.data[key] + other[key]
            if total > 0:
                out.data[key] = total
        for key in other.data:
            if key not in self.data and other.data[key] > 0:
                out.data[key] = other.data[key]
        return out

    def __sub__(self, other):
        out = Counter()
        for key in self.data:
            total = self.data[key] - other[key]
            if total > 0:
                out.data[key] = total
        for key in other.data:
            if key not in self.data and other.data[key] < 0:
                out.data[key] = -other.data[key]
        return out

    def __or__(self, other):
        out = Counter()
        for key in self.data:
            big = self.data[key]
            if other[key] > big:
                big = other[key]
            if big > 0:
                out.data[key] = big
        for key in other.data:
            if key not in self.data and other.data[key] > 0:
                out.data[key] = other.data[key]
        return out

    def __and__(self, other):
        out = Counter()
        for key in self.data:
            small = self.data[key]
            if other[key] < small:
                small = other[key]
            if small > 0:
                out.data[key] = small
        return out

    def __repr__(self):
        if len(self.data) == 0:
            return "Counter()"
        return "Counter(" + repr(dict(self.most_common())) + ")"


class deque:
    def __init__(self, iterable=None, maxlen=None):
        self.items = []
        self.head = 0
        self.maxlen = maxlen
        if iterable is not None:
            for item in iterable:
                self.append(item)

    def __len__(self):
        return len(self.items) - self.head

    def __bool__(self):
        return len(self.items) - self.head > 0

    def append(self, item):
        self.items.append(item)
        if self.maxlen is not None and len(self.items) - self.head > self.maxlen:
            self.popleft()

    def appendleft(self, item):
        if self.head == 0:
            pad = len(self.items) + 8
            self.items = [None] * pad + self.items
            self.head = pad
        self.head -= 1
        self.items[self.head] = item
        if self.maxlen is not None and len(self.items) - self.head > self.maxlen:
            self.pop()

    def pop(self):
        if len(self.items) - self.head == 0:
            raise IndexError("pop from an empty deque")
        return self.items.pop()

    def popleft(self):
        if len(self.items) - self.head == 0:
            raise IndexError("pop from an empty deque")
        item = self.items[self.head]
        self.items[self.head] = None
        self.head += 1
        if self.head > 32 and self.head * 2 > len(self.items):
            self.items = self.items[self.head:]
            self.head = 0
        return item

    def extend(self, iterable):
        for item in list(iterable):
            self.append(item)

    def extendleft(self, iterable):
        for item in list(iterable):
            self.appendleft(item)

    def clear(self):
        self.items = []
        self.head = 0

    def rotate(self, n=1):
        size = len(self.items) - self.head
        if size == 0:
            return
        k = n % size
        if k == 0:
            return
        live = self.items[self.head:]
        self.items = live[size - k:] + live[:size - k]
        self.head = 0

    def reverse(self):
        live = self.items[self.head:]
        live.reverse()
        self.items = live
        self.head = 0

    def copy(self):
        return deque(self.items[self.head:], self.maxlen)

    def count(self, value):
        total = 0
        for item in self.items[self.head:]:
            if item == value:
                total += 1
        return total

    def index(self, value):
        i = 0
        for item in self.items[self.head:]:
            if item == value:
                return i
            i += 1
        raise ValueError(str(value) + " is not in deque")

    def remove(self, value):
        live = self.items[self.head:]
        live.remove(value)
        self.items = live
        self.head = 0

    def __contains__(self, value):
        for item in self.items[self.head:]:
            if item == value:
                return True
        return False

    def __getitem__(self, i):
        size = len(self.items) - self.head
        if i < 0:
            i += size
        if i < 0 or i >= size:
            raise IndexError("deque index out of range")
        return self.items[self.head + i]

    def __setitem__(self, i, value):
        size = len(self.items) - self.head
        if i < 0:
            i += size
        if i < 0 or i >= size:
            raise IndexError("deque index out of range")
        self.items[self.head + i] = value

    def __iter__(self):
        for item in self.items[self.head:]:
            yield item

    def __eq__(self, other):
        if isinstance(other, deque):
            return self.items[self.head:] == other.items[other.head:]
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        live = repr(self.items[self.head:])
        if self.maxlen is not None:
            return "deque(" + live + ", maxlen=" + str(self.maxlen) + ")"
        return "deque(" + live + ")"
