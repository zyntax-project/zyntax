# `bytearray`: bytes that can change, held as a list of ints.


class bytearray:
    def __init__(self, source=None, encoding=None):
        self.data = []
        if source is None:
            return
        if isinstance(source, int):
            self.data = [0] * source
        elif isinstance(source, str):
            text = str(source)
            for b in text.encode("utf-8"):
                self.data.append(b)
        else:
            for b in source:
                self.data.append(b)

    def __bytes__(self):
        return bytes(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for b in self.data:
            yield b

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, value):
        self.data[i] = value

    def __contains__(self, value):
        return value in self.data

    def __eq__(self, other):
        if isinstance(other, bytearray):
            return self.data == other.data
        return bytes(self.data) == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        out = bytearray(self.data)
        for b in other:
            out.data.append(b)
        return out

    def __iadd__(self, other):
        for b in other:
            self.data.append(b)
        return self

    def __mul__(self, n):
        return bytearray(self.data * n)

    def __repr__(self):
        return "bytearray(" + repr(bytes(self.data)) + ")"

    def append(self, value):
        self.data.append(value)

    def extend(self, other):
        for b in other:
            self.data.append(b)

    def insert(self, i, value):
        self.data.insert(i, value)

    def pop(self, i=-1):
        return self.data.pop(i)

    def remove(self, value):
        self.data.remove(value)

    def clear(self):
        self.data.clear()

    def copy(self):
        return bytearray(self.data)

    def reverse(self):
        self.data.reverse()

    def decode(self, encoding="utf-8"):
        return bytes(self.data).decode()
